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
import math
import os
import pickle
import signal
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

from dq.game import GameState, draw_ply_limit
from dq.mcts import MCTS, Node, SearchConfig
from dq.model import (
    WDL_D, WDL_L, WDL_W, Example, ModelConfig, NetEvaluator, SquareTransformer, collate, load_model, loss_fn,
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
    # AlphaZero (Silver et al. 2018) scales alpha inversely with the move count: 0.3 chess (~35
    # moves), 0.15 shogi, 0.03 Go.  Brazilian draughts has ~7-8 legal moves => ~1.2.
    dirichlet_alpha: float = 1.2
    dirichlet_eps: float = 0.35
    contempt: float = 0.0               # Lc0-style search-time draw contempt in self-play (labels untouched)
    # decisive-start curriculum (checkers-rl): a fraction of games start from a random position
    # where one side has strictly more material, the strong colour chosen 50/50
    asym_start_frac: float = 0.0
    asym_weak: tuple[int, int] = (3, 9)       # weak side's piece count range
    asym_extra: int = 5                       # strong side has weak+1 .. weak+asym_extra (capped at 12)
    # regret buffer (checkers-rl): positions where |root Q - outcome| > regret_threshold are kept
    # per worker and a fraction of new games start from them
    regret_start_frac: float = 0.0
    regret_threshold: float = 0.5
    regret_buffer: int = 5000
    tree_reuse: bool = True
    endgame_start_frac: float = 0.0     # fraction of games started from a random endgame position (KataGo-style mixed starts)
    endgame_pieces: tuple[int, int] = (5, 8)   # total piece count range for those starts (must exceed tablebase coverage)
    tb_rescore: bool = False            # replace game-outcome value targets with exact tablebase WDL where covered (Lc0-style)
    tb_draw_rules: bool = True          # score tablebase wins/losses with DTM beyond the active draw rule's ply limit as draws
    # training
    replay_buffer: int = 500_000
    min_buffer: int = 5_000
    train_steps_per_round: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    lr_min: float = 1e-3                # cosine-decay floor; equal to lr => constant
    lr_decay_rounds: int = 0            # rounds over which lr decays to lr_min (0 => constant)
    weight_decay: float = 1e-4
    round_seconds: float = 60.0         # trainer drains the queue for this long per round
    # human-game auxiliary supervision (AlphaGo SL policy / Maia): each step adds a batch of
    # human positions; policy fits the human move, WDL fits the game result at reduced weight
    human_data: str | None = None       # pickle from scripts/human_sl.py build
    human_frac: float = 0.25            # human batch size as a fraction of batch_size
    human_wdl_weight: float = 0.5
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
        self.records: list[tuple[Position, dict, float]] = []     # (pos, policy target, root Q for the mover)


def tb_outcome(tb: Tablebase, pos: Position, draw_rules: bool = True) -> int | None:
    """Tablebase WDL for the side to move, or None if not covered.  With ``draw_rules`` a
    win/loss whose DTM exceeds the ply limit of the draw rule active at ``pos`` is a DRAW,
    because the game would be adjudicated first (lidraughts 7.2.4–7.2.8)."""
    if not tb.covers(pos):
        return None
    r = tb.probe(pos)
    if r is None:
        return None
    v, dtm = r
    if v != DRAW and draw_rules:
        limit = draw_ply_limit(pos)
        if limit is not None and dtm > limit:
            return DRAW
    return v


def _finish(g: _Game, tb: Tablebase | None = None, draw_rules: bool = True,
            regret_threshold: float | None = None) -> tuple[list[Example], list[Position]]:
    """Turn a finished game into training examples, plus the "regret" positions where the
    search's root value disagreed with the final (or tablebase) outcome by more than
    ``regret_threshold`` (checkers-rl's regret buffer; None disables).

    With ``tb``, positions the tablebase covers get the exact WDL for the side to move as
    their value target instead of the (possibly misplayed) game outcome — the "tablebase
    rescoring" used by Leela Chess Zero — discounted by the draw rules (:func:`tb_outcome`).
    """
    z_white = {1.0: 1.0, 0.5: 0.0, 0.0: -1.0}[g.state.result]
    out: list[Example] = []
    regret: list[Position] = []
    for p, pi, q in g.records:
        z = z_white if p.turn == WHITE else -z_white
        if tb is not None:
            v = tb_outcome(tb, p, draw_rules)
            if v is not None:
                z = float(v)             # WIN=1, DRAW=0, LOSS=-1, already side-to-move relative
        out.append(make_example(p, pi, z))
        if regret_threshold is not None and abs(q - z) > regret_threshold and (tb is None or not tb.covers(p)):
            regret.append(p)
    return out, regret


def _legal_start(pos: Position, tb: Tablebase | None) -> bool:
    return (tb is None or not tb.covers(pos)) and bool(legal_moves(pos)) and not GameState(pos).over


def _start_position(cfg: SelfPlayConfig, rng: random.Random, tb: Tablebase | None,
                    regret: "list[Position] | None" = None) -> Position:
    """Initial position for a new self-play game.  In order of precedence, each with its
    configured probability: a regret-buffer position (where search misjudged the outcome);
    a guaranteed-asymmetric material position (one side strictly ahead, strong colour 50/50
    — checkers-rl's curriculum, so the game is likely decisive); a random endgame above
    tablebase coverage; otherwise the standard start."""
    r = rng.random()
    if r < cfg.regret_start_frac:
        for _ in range(10 if regret else 0):
            pos = regret[rng.randrange(len(regret))]
            if _legal_start(pos, tb):
                return pos
        return Position.start()          # empty buffer (early in a run): the slot is a standard start
    r -= cfg.regret_start_frac
    if r < cfg.asym_start_frac:
        lo, hi = cfg.asym_weak
        for _ in range(50):
            weak = rng.randint(lo, hi)
            strong = rng.randint(weak + 1, min(12, weak + cfg.asym_extra))
            n_white, n_black = (strong, weak) if rng.random() < 0.5 else (weak, strong)
            pos = random_position(rng, n_white, n_black)
            if _legal_start(pos, tb):
                return pos
    r -= cfg.asym_start_frac
    if r < cfg.endgame_start_frac:
        lo, hi = cfg.endgame_pieces
        for _ in range(50):
            total = rng.randint(lo, hi)
            n_white = rng.randint(max(1, total - (total // 2 + 1)), min(total - 1, total // 2 + 1))
            pos = random_position(rng, n_white, total - n_white)
            if _legal_start(pos, tb):
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
    ev = NetEvaluator(model, dev, autocast=cfg.autocast, contempt=cfg.contempt)
    tb = Tablebase.load(tb_dir) if tb_dir else None
    sc = SearchConfig(simulations=cfg.simulations, dirichlet_alpha=cfg.dirichlet_alpha,
                      dirichlet_eps=cfg.dirichlet_eps, leaves_per_tree=cfg.leaves_per_tree)
    mcts = MCTS(ev, sc, tablebase=tb, rng=rng, tb_ply_limit=draw_ply_limit if cfg.tb_draw_rules else None)
    tb_score = tb if cfg.tb_rescore else None
    regret: list[Position] = []
    regret_thr = cfg.regret_threshold if cfg.regret_start_frac > 0 else None
    games = [_Game(mcts, _start_position(cfg, rng, tb)) for _ in range(cfg.games_per_worker)]
    t_last = time.time(); n_games = 0; n_moves = 0
    results = {"white": 0, "draw": 0, "black": 0}
    reasons: dict[str, int] = {}
    starts = {"standard": 0, "other": 0}

    parent = os.getppid()
    while not stop.is_set():
        if os.getppid() != parent:
            return                      # trainer died; don't run as an orphan
        mcts.search_batch([g.root for g in games], add_noise=True, seen=[g.state.seen for g in games])
        finished: list[Example] = []
        for i, g in enumerate(games):
            pi = mcts.policy_target(g.root)
            g.records.append((g.state.pos, pi, g.root.Q))
            t = 1.0 if len(g.state.history) < cfg.temperature_plies else cfg.temperature_late
            mv = mcts.select_move(g.root, t)
            g.state.push(mv)
            n_moves += 1
            if g.state.over or len(g.state.history) >= cfg.max_plies:
                if not g.state.over:
                    g.state.result, g.state.reason = 0.5, "max plies"
                exs, reg = _finish(g, tb_score, cfg.tb_draw_rules, regret_thr)
                finished.extend(exs)
                if reg:
                    regret.extend(reg)
                    del regret[:-cfg.regret_buffer]
                n_games += 1
                results[{1.0: "white", 0.5: "draw", 0.0: "black"}[g.state.result]] += 1
                reasons[g.state.reason] = reasons.get(g.state.reason, 0) + 1
                start = _start_position(cfg, rng, tb, regret)
                starts["standard" if start == Position.start() else "other"] += 1
                games[i] = _Game(mcts, start)
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
                                      "results": dict(results), "reasons": dict(reasons),
                                      "starts": dict(starts), "regret_buffer": len(regret)}))
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

def lr_at(cfg: SelfPlayConfig, rnd: int) -> float:
    """Cosine decay from ``lr`` to ``lr_min`` over ``lr_decay_rounds`` rounds, then flat."""
    if cfg.lr_decay_rounds <= 0 or cfg.lr_min >= cfg.lr:
        return cfg.lr
    t = min(rnd / cfg.lr_decay_rounds, 1.0)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * t))


def _human_loss(model, rows, cfg: SelfPlayConfig, device) -> tuple[torch.Tensor, float]:
    """Auxiliary loss on human positions: policy vs the human move (only where >1 legal move),
    WDL vs the game result at ``human_wdl_weight``.  ``rows`` are (Example, has_policy) pairs."""
    b = collate([e for e, _ in rows], device)
    has_pol = torch.tensor([float(p) for _, p in rows], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.autocast and device.type == "cuda"):
        out = model(b["ids"])
    logp = torch.log_softmax(out["policy"].float(), -1)
    l_pol = (-(b["policy"] * logp).sum(-1) * has_pol).sum() / has_pol.sum().clamp(min=1)
    l_wdl = torch.nn.functional.cross_entropy(out["wdl"].float(), b["wdl"])
    return l_pol + cfg.human_wdl_weight * l_wdl, l_pol.item()


def train_steps(model, opt, buffer: list[Example], cfg: SelfPlayConfig, device, rng: random.Random,
                human: list | None = None) -> dict[str, float]:
    model.train()
    agg: dict[str, float] = {}
    n_h = int(cfg.batch_size * cfg.human_frac) if human else 0
    for _ in range(cfg.train_steps_per_round):
        batch = collate(rng.sample(buffer, min(cfg.batch_size, len(buffer))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.autocast and device.type == "cuda"):
            out = model(batch["ids"])
        loss, parts = loss_fn({k: v.float() for k, v in out.items()}, batch)
        if n_h:
            hl, hp = _human_loss(model, rng.sample(human, min(n_h, len(human))), cfg, device)
            loss = loss + hl
            parts["human_policy"] = hp
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
    ma = MCTS(ev_a, sc, tablebase=tb, rng=rng, tb_ply_limit=draw_ply_limit)
    mb = MCTS(ev_b, sc, tablebase=tb, rng=rng, tb_ply_limit=draw_ply_limit)
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
            m.search_batch(rs, add_noise=False, seen=[states[i].seen for i in idx])
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
    wrong = sum(1 for a, p in zip(pred, positions) if a != tmap[tb_outcome(tb, p)])
    return wrong / n


def gate(model, prev, cfg: SelfPlayConfig, tb, device, rng, anchor=None) -> dict[str, float]:
    ev, ev_prev = NetEvaluator(model, device), NetEvaluator(prev, device)
    out = {"vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies),
           # raw policy vs the previous checkpoint *with* eval_sims search: a hard bar, expected low
           "raw_vs_prev": match(ev, ev_prev, cfg.eval_games, cfg.eval_sims, tb, rng, cfg.max_plies, raw_a=True),
           # raw policy vs the previous checkpoint at 1 node: the PLAN Phase 1 freeze criterion (>0.80)
           "raw_vs_1node": match(ev, ev_prev, cfg.eval_games, 1, tb, rng, cfg.max_plies, raw_a=True)}
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


def _save_buffer(buffer: list[Example], path: Path) -> None:
    """Checkpoint the replay buffer (written atomically) so a restart doesn't start from empty."""
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load_buffer(path: Path) -> list[Example]:
    with open(path, "rb") as f:
        return pickle.load(f)


def run(out_dir: str | Path, rounds: int, cfg: SelfPlayConfig, mcfg: ModelConfig | None = None,
        tb_dir: str | None = None, device: str | None = None, resume: str | None = None,
        resume_buffer: str | None = None) -> None:
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
        anchor = load_model(cfg.anchor, device_t)        # may be a different size than the candidate
        print(f"anchor {cfg.anchor}: {anchor.n_params():,} params", flush=True)
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
    if resume_buffer:
        buffer = _load_buffer(Path(resume_buffer))[-cfg.replay_buffer:]
        print(f"resumed replay buffer: {len(buffer):,} examples from {resume_buffer}", flush=True)
    stats: dict[int, dict] = {}
    n_champ = 0
    human = None
    if cfg.human_data:
        with open(cfg.human_data, "rb") as f:
            human = pickle.load(f)["train"]
        print(f"human auxiliary data: {len(human):,} positions, {cfg.human_frac:.0%} of each batch", flush=True)
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
            lr_now = lr_at(cfg, rnd)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            losses = train_steps(model, opt, buffer, cfg, device_t, rng, human)
            t_train = time.time() - t0 - t_gen
            rec = {"round": rnd, "version": version, "new_games": n_games, "new_examples": n_new,
                   "buffer": len(buffer), "loss": losses, "lr": lr_now, "t_gen": round(t_gen, 1),
                   "t_train": round(t_train, 1), "worker_stats": stats}
            if (rnd + 1) % cfg.gate_every == 0:
                ev_out = gate(model, prev, cfg, tb, device_t, rng, anchor)
                promoted = ev_out["vs_prev"] >= cfg.promote_threshold
                rec["gate"] = ev_out; rec["promoted"] = promoted
                # champion-style reference: `prev` moves only when the candidate beats it, so a
                # failed gate lets progress accumulate and the next gate measures a longer span.
                # Training itself continues from the live weights unless --gate-revert.
                if promoted:
                    prev.load_state_dict(model.state_dict())
                    n_champ += 1
                    torch.save(model.state_dict(), out / f"champion_{n_champ:03d}_r{rnd + 1}.pt")
                    rec["champion"] = n_champ
                elif cfg.gate_revert:
                    model.load_state_dict(prev.state_dict())
                    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                version += 1
                _publish(model, weights_dir, version)
                torch.save(model.state_dict(), out / f"gen_{version:03d}.pt")
                _save_buffer(buffer, out / "buffer.pkl")
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
