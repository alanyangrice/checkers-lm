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

import json
import os
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
    tb_samples: int = 2000
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


def _finish(g: _Game) -> list[Example]:
    z_white = {1.0: 1.0, 0.5: 0.0, 0.0: -1.0}[g.state.result]
    return [make_example(p, pi, z_white if p.turn == WHITE else -z_white) for p, pi in g.records]


def worker_loop(wid: int, cfg: SelfPlayConfig, mcfg: ModelConfig, weights_dir: str, out_q: mp.Queue,
                stop: mp.Event, tb_dir: str | None, device: str) -> None:
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
    games = [_Game(mcts, Position.start()) for _ in range(cfg.games_per_worker)]
    t_last = time.time(); n_games = 0; n_moves = 0
    results = {"white": 0, "draw": 0, "black": 0}
    reasons: dict[str, int] = {}

    while not stop.is_set():
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
                finished.extend(_finish(g))
                n_games += 1
                results[{1.0: "white", 0.5: "draw", 0.0: "black"}[g.state.result]] += 1
                reasons[g.state.reason] = reasons.get(g.state.reason, 0) + 1
                games[i] = _Game(mcts, Position.start())
            else:
                g.root = MCTS.advance(g.root, mv) if cfg.tree_reuse else Node(g.state.pos)
        if finished:
            out_q.put((wid, version, finished))
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
          max_plies: int = 300, raw_a: bool = False) -> float:
    """Score of A over ``games`` games (alternating colours), all played
    concurrently: each side's searches are batched across the games it is
    to move in.  ``raw_a`` makes A play its raw policy (no search)."""
    sc = SearchConfig(simulations=sims, dirichlet_eps=0.0, leaves_per_tree=8)
    ma, mb = MCTS(ev_a, sc, tablebase=tb, rng=rng), MCTS(ev_b, sc, tablebase=tb, rng=rng)
    states = [GameState(Position.start()) for _ in range(games)]
    a_is_white = [i % 2 == 0 for i in range(games)]
    roots: list[Node | None] = [None] * games
    while True:
        live = [i for i, g in enumerate(states) if not g.over and len(g.history) < max_plies]
        if not live:
            break
        for side_a in (True, False):
            idx = [i for i in live if (states[i].pos.turn == WHITE) == (a_is_white[i] == side_a)]
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
            rs = []
            for i in idx:
                if roots[i] is None or roots[i].pos != states[i].pos:
                    roots[i] = Node(states[i].pos)
                rs.append(roots[i])
            m.search_batch(rs, add_noise=False)
            for i, r in zip(idx, rs):
                mv = m.select_move(r, 0.0)
                states[i].push(mv)
                roots[i] = MCTS.advance(r, mv)
    score = 0.0
    for g, aw in zip(states, a_is_white):
        res = g.result if g.result is not None else 0.5
        score += res if aw else 1.0 - res
    return score / games


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


def gate(model, prev, cfg: SelfPlayConfig, tb, device, rng) -> dict[str, float]:
    ev, ev_prev = NetEvaluator(model, device), NetEvaluator(prev, device)
    out = {"vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies),
           "raw_vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies, raw_a=True)}
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
            t0 = time.time(); n_new = 0; n_games = 0
            while time.time() - t0 < cfg.round_seconds or len(buffer) < cfg.min_buffer:
                try:
                    wid, ver, payload = q.get(timeout=1.0)
                except Exception:
                    if not any(p.is_alive() for p in procs):
                        raise RuntimeError("all workers died")
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
                ev_out = gate(model, prev, cfg, tb, device_t, rng)
                promoted = ev_out["vs_prev"] >= cfg.promote_threshold
                rec["gate"] = ev_out; rec["promoted"] = promoted
                if promoted:
                    prev.load_state_dict(model.state_dict())
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
