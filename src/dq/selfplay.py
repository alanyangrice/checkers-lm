"""AlphaZero-style self-play (PLAN Phase 1).

Architecture
------------
::

    trainer (main process, GPU)                 worker 0..W-1 (GPU, own CUDA context)
    ┌─────────────────────────────┐             ┌─────────────────────────────────┐
    │ replay buffer (FIFO)        │  examples   │ G concurrent games              │
    │ train K steps / round       │ ◀────mp.Q───│ MCTS.search_batch over G trees  │
    │ gate vs previous checkpoint │             │  → 1 forward pass per round     │
    │ publish weights.pt + version│ ──file────▶ │ reload weights at game boundary │
    └─────────────────────────────┘             └─────────────────────────────────┘

Differences from the classic "CPU workers + GPU inference server" layout
(e.g. alanyangrice/checkers-rl): there is no per-leaf IPC.  Each worker
batches leaves across its own G games with virtual loss and calls the
network in-process, so batch size is ``G × leaves_per_tree`` instead of
"number of workers", and the only traffic between processes is finished
games.  Several workers share one GPU; a 5M-param model on 32 tokens at
batch 256 is ~1 ms on a 5090, so the Python tree code is the bottleneck
and more workers scale it until the CPU is saturated.

Gates (PLAN Phase 1), computed by the trainer every ``gate_every`` rounds:
* ``vs_prev``        search vs search against the last promoted checkpoint
* ``raw_vs_prev``    raw policy (no search) vs previous checkpoint's search
* ``tb_wdl_err``     WDL-head error rate on random tablebase positions
"""
from __future__ import annotations

import faulthandler
import json
import os
import signal
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

from dq.game import GameState
from dq.mcts import MCTS, Node, SearchConfig
from dq.model import (
    WDL_D, WDL_L, WDL_W, Example, ModelConfig, NetEvaluator, SquareTransformer, collate, loss_fn,
    make_example,
)
from dq.rules import WHITE, Position, legal_moves, random_position
from dq.tablebase import DRAW, LOSS, WIN, Tablebase
from dq.tokens import encode_board


@dataclass
class SelfPlayConfig:
    # generation
    workers: int = 4
    games_per_worker: int = 32          # concurrent trees per worker
    simulations: int = 200
    leaves_per_tree: int = 8
    temperature_plies: int = 20
    temperature_late: float = 0.25
    max_plies: int = 300
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    tree_reuse: bool = True
    endgame_start_frac: float = 0.0     # fraction of games started from a random endgame position (KataGo-style mixed starts)
    endgame_pieces: tuple[int, int] = (5, 8)   # total piece count range for those starts (must exceed tablebase coverage)
    tb_rescore: bool = False            # replace game-outcome value targets with exact tablebase WDL where covered (Lc0-style)
    # training
    replay_buffer: int = 500_000
    min_buffer: int = 5_000
    train_steps_per_round: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    round_seconds: float = 60.0         # trainer drains the queue for this long per round
    # gating
    gate_every: int = 5
    eval_games: int = 40
    eval_sims: int = 100
    promote_threshold: float = 0.55
    gate_revert: bool = False           # AlphaGo-Zero style: reject and revert on a failed gate
    tb_samples: int = 2000
    anchor: str | None = None           # checkpoint path; each gate also plays it (vs_anchor) as a fixed reference
    anchor_games: int = 100
    # misc
    seed: int = 0
    autocast: bool = True


# ---------------------------------------------------------------------------
# worker: G concurrent games, batched leaf evaluation
# ---------------------------------------------------------------------------

class _Game:
    __slots__ = ("state", "root", "records")

    def __init__(self, mcts: MCTS, start: Position) -> None:
        self.state = GameState(start)
        self.root = Node(start)
        self.records: list[tuple[Position, dict]] = []


def _finish(g: _Game, tb: Tablebase | None = None) -> list[Example]:
    """Turn a finished game into training examples.

    With ``tb``, positions the tablebase covers get the exact WDL for the side to move as
    their value target instead of the (possibly misplayed) game outcome — the "tablebase
    rescoring" used by Leela Chess Zero.
    """
    z_white = {1.0: 1.0, 0.5: 0.0, 0.0: -1.0}[g.state.result]
    out: list[Example] = []
    for p, pi in g.records:
        z = z_white if p.turn == WHITE else -z_white
        if tb is not None and tb.covers(p):
            r = tb.probe(p)
            if r is not None:
                z = float(r[0])          # WIN=1, DRAW=0, LOSS=-1, already side-to-move relative
        out.append(make_example(p, pi, z))
    return out


def _start_position(cfg: SelfPlayConfig, rng: random.Random, tb: Tablebase | None) -> Position:
    """Initial position for a new self-play game: the standard start, or (with probability
    ``cfg.endgame_start_frac``) a random legal endgame with a piece count above tablebase
    coverage, so search can resolve it exactly while the net still has to judge it."""
    if cfg.endgame_start_frac <= 0 or rng.random() >= cfg.endgame_start_frac:
        return Position.start()
    lo, hi = cfg.endgame_pieces
    for _ in range(50):
        total = rng.randint(lo, hi)
        n_white = rng.randint(max(1, total - (total // 2 + 1)), min(total - 1, total // 2 + 1))
        pos = random_position(rng, n_white, total - n_white)
        if (tb is None or not tb.covers(pos)) and legal_moves(pos) and not GameState(pos).over:
            return pos
    return Position.start()


def _register_stack_dump() -> None:
    """On POSIX, `kill -USR1 <pid>` dumps all thread stacks to stderr. No-op on Windows."""
    sig = getattr(signal, "SIGUSR1", None)
    if sig is not None:
        faulthandler.register(sig, all_threads=True)


def worker_loop(wid: int, cfg: SelfPlayConfig, mcfg: ModelConfig, weights_dir: str, out_q: mp.Queue,
                stop: mp.Event, tb_dir: str | None, device: str) -> None:
    _register_stack_dump()
    torch.set_num_threads(1)
    rng = random.Random(cfg.seed * 1000 + wid)
    torch.manual_seed(cfg.seed * 1000 + wid)
    dev = torch.device(device)
    model = SquareTransformer(mcfg).to(dev)
    version = _load_latest(model, weights_dir, -1, dev)
    ev = NetEvaluator(model, dev, autocast=cfg.autocast)
    tb = Tablebase.load(tb_dir) if tb_dir else None
    sc = SearchConfig(simulations=cfg.simulations, dirichlet_alpha=cfg.dirichlet_alpha,
                      dirichlet_eps=cfg.dirichlet_eps, leaves_per_tree=cfg.leaves_per_tree)
    mcts = MCTS(ev, sc, tablebase=tb, rng=rng)
    tb_score = tb if cfg.tb_rescore else None
    games = [_Game(mcts, _start_position(cfg, rng, tb)) for _ in range(cfg.games_per_worker)]
    t_last = time.time(); n_games = 0; n_moves = 0
    results = {"white": 0, "draw": 0, "black": 0}
    reasons: dict[str, int] = {}

    parent = os.getppid()
    while not stop.is_set():
        if os.getppid() != parent:
            return                      # trainer died; don't run as an orphan
        mcts.search_batch([g.root for g in games], add_noise=True)
        finished: list[Example] = []
        for i, g in enumerate(games):
            pi = mcts.policy_target(g.root)
            g.records.append((g.state.pos, pi))
            t = 1.0 if len(g.state.history) < cfg.temperature_plies else cfg.temperature_late
            mv = mcts.select_move(g.root, t)
            g.state.push(mv)
            n_moves += 1
            if g.state.over or len(g.state.history) >= cfg.max_plies:
                if not g.state.over:
                    g.state.result, g.state.reason = 0.5, "max plies"
                finished.extend(_finish(g, tb_score))
                n_games += 1
                results[{1.0: "white", 0.5: "draw", 0.0: "black"}[g.state.result]] += 1
                reasons[g.state.reason] = reasons.get(g.state.reason, 0) + 1
                games[i] = _Game(mcts, _start_position(cfg, rng, tb))
            else:
                g.root = MCTS.advance(g.root, mv) if cfg.tree_reuse else Node(g.state.pos)
        if finished:
            try:
                out_q.put((wid, version, finished), timeout=60)
            except Exception:
                pass                        # trainer unresponsive; drop the games rather than wedge
            # pick up new weights between games
            version = _load_latest(model, weights_dir, version, dev)
        if time.time() - t_last > 30:
            out_q.put((wid, version, {"stat": True, "games": n_games, "moves": n_moves,
                                      "evals": ev.calls, "positions": ev.positions,
                                      "results": dict(results), "reasons": dict(reasons)}))
            t_last = time.time()


def _load_latest(model: SquareTransformer, weights_dir: str, have: int, dev) -> int:
    vfile = Path(weights_dir) / "version"
    try:
        v = int(vfile.read_text())
    except (FileNotFoundError, ValueError):
        return have
    if v <= have:
        return have
    for _ in range(5):
        try:
            model.load_state_dict(torch.load(Path(weights_dir) / "latest.pt", map_location=dev))
            model.eval()
            return v
        except Exception:
            time.sleep(0.2)
    return have


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

def train_steps(model, opt, buffer: list[Example], cfg: SelfPlayConfig, device, rng: random.Random) -> dict[str, float]:
    model.train()
    agg: dict[str, float] = {}
    for _ in range(cfg.train_steps_per_round):
        batch = collate(rng.sample(buffer, min(cfg.batch_size, len(buffer))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.autocast and device.type == "cuda"):
            out = model(batch["ids"])
        loss, parts = loss_fn({k: v.float() for k, v in out.items()}, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
    model.eval()
    n = max(cfg.train_steps_per_round, 1)
    return {k: v / n for k, v in agg.items()}


def match(ev_a: NetEvaluator, ev_b: NetEvaluator, games: int, sims: int, tb, rng: random.Random,
          max_plies: int = 300, raw_a: bool = False, opening_plies: int = 6) -> float:
    """Score of A over ``games`` games, all played concurrently: each side's
    searches are batched across the games it is to move in.  Search is
    deterministic (T=0, no noise), so every pair of games starts from the
    same random ``opening_plies``-ply opening played once with each colour.
    ``raw_a`` makes A play its raw policy (no search)."""
    sc = SearchConfig(simulations=sims, dirichlet_eps=0.0, leaves_per_tree=8)
    ma, mb = MCTS(ev_a, sc, tablebase=tb, rng=rng), MCTS(ev_b, sc, tablebase=tb, rng=rng)
    states = []
    for i in range(games):
        if i % 2 == 0:
            g = GameState(Position.start())
            for _ in range(opening_plies):
                if g.over:
                    break
                g.push(rng.choice(g.legal()))
            opening = [mv for _, mv in g.history]
        g = GameState(Position.start())
        for mv in opening:
            g.push(mv)
        states.append(g)
    a_is_white = [i % 2 == 0 for i in range(games)]
    # one tree per *player* per game: sharing a tree would hand each side the other's priors and
    # values and blend the two networks (compressing every match score toward 0.5)
    roots_by_side: dict[bool, list[Node | None]] = {True: [None] * games, False: [None] * games}
    while True:
        live = [i for i, g in enumerate(states) if not g.over and len(g.history) < max_plies]
        if not live:
            break
        for side_a in (True, False):
            idx = [i for i in live if not states[i].over
                   and (states[i].pos.turn == WHITE) == (a_is_white[i] == side_a)]
            if not idx:
                continue
            m = ma if side_a else mb
            if side_a and raw_a:
                for i in idx:
                    g = states[i]
                    moves = g.legal()
                    (prior, _), = ev_a([g.pos], [moves])
                    g.push(max(moves, key=lambda mv: prior[mv]))
                continue
            roots = roots_by_side[side_a]
            rs = []
            for i in idx:
                r = roots[i]
                if r is not None and len(states[i].history) >= 1:
                    r = MCTS.advance(r, states[i].history[-1][1])     # follow the opponent's reply
                if r is None or r.pos != states[i].pos:
                    r = Node(states[i].pos)
                rs.append(r)
            m.search_batch(rs, add_noise=False)
            for i, r in zip(idx, rs):
                mv = m.select_move(r, 0.0)
                states[i].push(mv)
                roots[i] = MCTS.advance(r, mv)
    score = 0.0
    for g, aw in zip(states, a_is_white):
        res = g.result if g.result is not None else 0.5
        score += res if aw else 1.0 - res
    match.last_wdl = _wdl_counts(states, a_is_white)
    return score / games


def _wdl_counts(states, a_is_white) -> tuple[int, int, int]:
    """(wins, draws, losses) for A, for diagnostics."""
    w = d = l = 0
    for g, aw in zip(states, a_is_white):
        res = g.result if g.result is not None else 0.5
        s = res if aw else 1.0 - res
        if s > 0.5: w += 1
        elif s < 0.5: l += 1
        else: d += 1
    return w, d, l


def raw_policy(ev: NetEvaluator):
    def policy(pos, moves):
        (prior, _), = ev([pos], [moves])
        return max(moves, key=lambda m: prior[m])
    return policy


@torch.no_grad()
def tablebase_wdl_error(model, tb: Tablebase, n: int, rng: random.Random, device) -> float:
    positions = []
    while len(positions) < n:
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if tb.covers(p) and legal_moves(p):
            positions.append(p)
    ids = torch.tensor([encode_board(p) for p in positions], dtype=torch.long, device=device)
    pred = model(ids)["wdl"].argmax(-1).tolist()
    tmap = {WIN: WDL_W, DRAW: WDL_D, LOSS: WDL_L}
    wrong = sum(1 for a, p in zip(pred, positions) if a != tmap[tb.probe(p)[0]])
    return wrong / n


def gate(model, prev, cfg: SelfPlayConfig, tb, device, rng, anchor=None) -> dict[str, float]:
    ev, ev_prev = NetEvaluator(model, device), NetEvaluator(prev, device)
    out = {"vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies),
           "raw_vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies, raw_a=True)}
    if anchor is not None:
        # fixed reference checkpoint: a trend line, where vs_prev (5 rounds apart) is mostly noise
        out["vs_anchor"] = match(ev, NetEvaluator(anchor, device), cfg.anchor_games, cfg.eval_sims, tb, rng, cfg.max_plies)
        out["vs_anchor_wdl"] = list(match.last_wdl)
    if tb is not None:
        out["tb_wdl_err"] = tablebase_wdl_error(model, tb, cfg.tb_samples, rng, device)
    return out


def _publish(model, weights_dir: Path, version: int) -> None:
    tmp = weights_dir / "latest.pt.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, weights_dir / "latest.pt")
    (weights_dir / "version").write_text(str(version))


def run(out_dir: str | Path, rounds: int, cfg: SelfPlayConfig, mcfg: ModelConfig | None = None,
        tb_dir: str | None = None, device: str | None = None, resume: str | None = None) -> None:
    _register_stack_dump()   # kill -USR1 <pid> dumps all stacks to stderr (POSIX only)
    mcfg = mcfg or ModelConfig()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    weights_dir = out / "weights"; weights_dir.mkdir(exist_ok=True)
    dev_name = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    device_t = torch.device(dev_name)
    rng = random.Random(cfg.seed); torch.manual_seed(cfg.seed)

    model = SquareTransformer(mcfg).to(device_t)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device_t))
    model.eval()
    version = 0
    _publish(model, weights_dir, version)
    prev = SquareTransformer(mcfg).to(device_t); prev.load_state_dict(model.state_dict()); prev.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tb = Tablebase.load(tb_dir) if tb_dir else None
    anchor = None
    if cfg.anchor:
        anchor = SquareTransformer(mcfg).to(device_t)
        anchor.load_state_dict(torch.load(cfg.anchor, map_location=device_t)); anchor.eval()
    (out / "config.json").write_text(json.dumps({"selfplay": asdict(cfg), "model": asdict(mcfg)}, indent=1))
    log = open(out / "log.jsonl", "a")
    print(f"device={dev_name} params={model.n_params():,} workers={cfg.workers}", flush=True)

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue(maxsize=10_000)
    stop = ctx.Event()
    procs = [ctx.Process(target=worker_loop, args=(w, cfg, mcfg, str(weights_dir), q, stop, tb_dir, dev_name),
                         daemon=True) for w in range(cfg.workers)]
    for p in procs:
        p.start()

    buffer: list[Example] = []
    stats: dict[int, dict] = {}
    try:
        for rnd in range(rounds):
            t0 = time.time(); n_new = 0; n_games = 0; t_last_item = time.time()
            while time.time() - t0 < cfg.round_seconds or len(buffer) < cfg.min_buffer:
                try:
                    wid, ver, payload = q.get(timeout=1.0)
                    t_last_item = time.time()
                except Exception:
                    if not any(p.is_alive() for p in procs):
                        raise RuntimeError("all workers died")
                    if time.time() - t_last_item > 120:
                        print(json.dumps({"warn": "no games received for 120s", "qsize": q.qsize(),
                                          "alive": [p.is_alive() for p in procs], "stats": stats}), flush=True)
                        t_last_item = time.time()
                    continue
                if isinstance(payload, dict):
                    stats[wid] = payload
                    continue
                buffer.extend(payload); n_new += len(payload); n_games += 1
            buffer = buffer[-cfg.replay_buffer:]
            t_gen = time.time() - t0
            losses = train_steps(model, opt, buffer, cfg, device_t, rng)
            t_train = time.time() - t0 - t_gen
            rec = {"round": rnd, "version": version, "new_games": n_games, "new_examples": n_new,
                   "buffer": len(buffer), "loss": losses, "t_gen": round(t_gen, 1), "t_train": round(t_train, 1),
                   "worker_stats": stats}
            if (rnd + 1) % cfg.gate_every == 0:
                ev_out = gate(model, prev, cfg, tb, device_t, rng, anchor)
                promoted = ev_out["vs_prev"] >= cfg.promote_threshold
                rec["gate"] = ev_out; rec["promoted"] = promoted
                if promoted or not cfg.gate_revert:
                    prev.load_state_dict(model.state_dict())      # new reference for the next gate
                    version += 1
                    _publish(model, weights_dir, version)
                    torch.save(model.state_dict(), out / f"gen_{version:03d}.pt")
                else:
                    model.load_state_dict(prev.state_dict())
                    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            else:
                # publish continuously so workers track the live network (AlphaZero-style)
                version += 1
                _publish(model, weights_dir, version)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
    finally:
        stop.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
