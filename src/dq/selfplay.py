"""AlphaZero-style self-play: generate -> train -> evaluate (PLAN Phase 1).

Single-process reference implementation.  It is correct and simple; the
throughput work (batched leaf evaluation, multiprocess game generation)
comes after the loop is shown to learn.

Gates (PLAN §5 Phase 1), all in :func:`evaluate`:
* ``vs_prev``      — score against the previous checkpoint (promote if > 0.55)
* ``vs_1node``     — raw policy (1-node) vs full search with the same net;
                     the net is "good enough to freeze" when raw play beats
                     the *previous* generation's search
* ``tb_wdl_err``   — fraction of random tablebase positions whose WDL the net
                     gets wrong (argmax of the WDL head vs exact value)
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dq.game import GameState, play_game
from dq.mcts import MCTS, SearchConfig, mcts_policy
from dq.model import (
    WDL_D, WDL_L, WDL_W, Example, ModelConfig, NetEvaluator, SquareTransformer, collate, loss_fn,
    make_example,
)
from dq.rules import BLACK, WHITE, Position, legal_moves, random_position
from dq.tablebase import DRAW, LOSS, WIN, Tablebase
from dq.tokens import encode_board


@dataclass
class SelfPlayConfig:
    games_per_iter: int = 50
    simulations: int = 100
    temperature_plies: int = 20        # sample with T=1 for the first N plies, then greedy
    max_plies: int = 300
    replay_buffer: int = 50_000        # examples kept
    train_steps: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    eval_games: int = 20
    eval_sims: int = 50
    promote_threshold: float = 0.55
    tb_samples: int = 2000
    seed: int = 0


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def generate_game(mcts: MCTS, cfg: SelfPlayConfig, rng: random.Random,
                  start: Position | None = None) -> tuple[list[Example], GameState]:
    g = GameState(start or Position.start())
    records: list[tuple[Position, dict]] = []
    while not g.over and len(g.history) < cfg.max_plies:
        root = mcts.search(g.pos, add_noise=True)
        pi = mcts.policy_target(root)
        records.append((g.pos, pi))
        t = 1.0 if len(g.history) < cfg.temperature_plies else 0.0
        g.push(mcts.select_move(root, t))
    if not g.over:
        g.result, g.reason = 0.5, "max plies"
    z_white = {1.0: 1.0, 0.5: 0.0, 0.0: -1.0}[g.result]
    exs = [make_example(p, pi, z_white if p.turn == WHITE else -z_white) for p, pi in records]
    return exs, g


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(model: SquareTransformer, buffer: list[Example], cfg: SelfPlayConfig, device,
          rng: random.Random, opt: torch.optim.Optimizer | None = None) -> dict[str, float]:
    model.train()
    opt = opt or torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    agg: dict[str, float] = {}
    n = 0
    for _ in range(cfg.train_steps):
        batch = collate(rng.sample(buffer, min(cfg.batch_size, len(buffer))), device)
        loss, parts = loss_fn(model(batch["ids"]), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    model.eval()
    return {k: v / max(n, 1) for k, v in agg.items()}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def match(pol_a, pol_b, games: int, rng: random.Random, max_plies: int = 300) -> float:
    """Score of A over `games` games, alternating colours."""
    score = 0.0
    for i in range(games):
        if i % 2 == 0:
            score += play_game(pol_a, pol_b, max_plies=max_plies).result
        else:
            score += 1.0 - play_game(pol_b, pol_a, max_plies=max_plies).result
    return score / games


def raw_policy(ev: NetEvaluator, rng: random.Random, greedy: bool = True):
    def policy(pos, moves):
        prior, _ = ev(pos, moves)
        if greedy:
            return max(moves, key=lambda m: prior[m])
        return rng.choices(moves, weights=[prior[m] for m in moves])[0]
    return policy


@torch.no_grad()
def tablebase_wdl_error(model: SquareTransformer, tb: Tablebase, n: int, rng: random.Random, device) -> float:
    positions = []
    while len(positions) < n:
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if tb.covers(p) and legal_moves(p):
            positions.append(p)
    ids = torch.tensor([encode_board(p) for p in positions], dtype=torch.long, device=device)
    pred = model(ids)["wdl"].argmax(-1).tolist()
    truth = [tb.probe(p)[0] for p in positions]
    tmap = {WIN: WDL_W, DRAW: WDL_D, LOSS: WDL_L}
    wrong = sum(1 for a, b in zip(pred, truth) if a != tmap[b])
    return wrong / n


def evaluate(model: SquareTransformer, prev: SquareTransformer | None, cfg: SelfPlayConfig,
             tb: Tablebase | None, device, rng: random.Random) -> dict[str, float]:
    ev = NetEvaluator(model, device)
    sc = SearchConfig(simulations=cfg.eval_sims, dirichlet_eps=0.0)
    out: dict[str, float] = {}
    if prev is not None:
        ev_prev = NetEvaluator(prev, device)
        out["vs_prev"] = match(mcts_policy(MCTS(ev, sc, tablebase=tb, rng=rng)),
                               mcts_policy(MCTS(ev_prev, sc, tablebase=tb, rng=rng)),
                               cfg.eval_games, rng, cfg.max_plies)
        out["raw_vs_prev_search"] = match(raw_policy(ev, rng),
                                          mcts_policy(MCTS(ev_prev, sc, tablebase=tb, rng=rng)),
                                          cfg.eval_games, rng, cfg.max_plies)
    if tb is not None:
        out["tb_wdl_err"] = tablebase_wdl_error(model, tb, cfg.tb_samples, rng, device)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(out_dir: str | Path, iters: int, cfg: SelfPlayConfig, mcfg: ModelConfig | None = None,
        tb_dir: str | None = None, device: str | None = None, resume: str | None = None) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                     "mps" if torch.backends.mps.is_available() else "cpu"))
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = SquareTransformer(mcfg).to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
    model.eval()
    tb = Tablebase.load(tb_dir) if tb_dir else None
    buffer: list[Example] = []
    log = open(out / "log.jsonl", "a")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    print(f"device={device} params={model.n_params():,}", flush=True)

    for it in range(iters):
        t0 = time.time()
        ev = NetEvaluator(model, device)
        mcts = MCTS(ev, SearchConfig(simulations=cfg.simulations), tablebase=tb, rng=rng)
        results = {"1.0": 0, "0.5": 0, "0.0": 0}
        plies = 0
        for _ in range(cfg.games_per_iter):
            exs, g = generate_game(mcts, cfg, rng)
            buffer.extend(exs)
            results[str(g.result)] += 1
            plies += len(g.history)
        buffer = buffer[-cfg.replay_buffer:]
        t_gen = time.time() - t0

        prev = SquareTransformer(model.cfg).to(device)
        prev.load_state_dict(model.state_dict()); prev.eval()
        losses = train(model, buffer, cfg, device, rng, opt)
        t_train = time.time() - t0 - t_gen

        ev_out = evaluate(model, prev, cfg, tb, device, rng)
        promoted = ev_out.get("vs_prev", 1.0) >= cfg.promote_threshold
        if not promoted:
            model.load_state_dict(prev.state_dict())
        torch.save(model.state_dict(), out / f"gen_{it:03d}.pt")
        torch.save(model.state_dict(), out / "latest.pt")
        rec = {"iter": it, "games": results, "avg_plies": plies / cfg.games_per_iter, "buffer": len(buffer),
               "loss": losses, "eval": ev_out, "promoted": promoted,
               "t_gen": round(t_gen, 1), "t_train": round(t_train, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
