"""Run self-play training.

Smoke test (Mac, a few minutes):
    .venv/bin/python scripts/selfplay.py runs/smoke --iters 2 --games 4 --sims 16 \
        --train-steps 30 --eval-games 2 --eval-sims 8 --small

Real run (5090):
    .venv/bin/python scripts/selfplay.py runs/gen1 --iters 200 --games 200 --sims 200 \
        --train-steps 1000 --tb data/tb
"""
from __future__ import annotations

import argparse

from dq.model import ModelConfig
from dq.selfplay import SelfPlayConfig, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--eval-sims", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--tb", default=None, help="tablebase directory (data/tb)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--small", action="store_true", help="tiny model for smoke tests")
    a = ap.parse_args()
    cfg = SelfPlayConfig(games_per_iter=a.games, simulations=a.sims, train_steps=a.train_steps,
                         batch_size=a.batch, lr=a.lr, eval_games=a.eval_games, eval_sims=a.eval_sims,
                         max_plies=a.max_plies, seed=a.seed)
    mcfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128) if a.small else ModelConfig()
    run(a.out, a.iters, cfg, mcfg, tb_dir=a.tb, device=a.device, resume=a.resume)


if __name__ == "__main__":
    main()
