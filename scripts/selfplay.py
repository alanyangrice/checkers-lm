"""Run self-play training (see dq/selfplay.py for the architecture).

Smoke test (Mac, ~2 min):
    .venv/bin/python scripts/selfplay.py runs/smoke --rounds 3 --workers 2 --games-per-worker 8 \
        --sims 32 --round-seconds 20 --min-buffer 500 --train-steps 50 --gate-every 2 \
        --eval-games 4 --eval-sims 16 --small --tb data/tb

5090 (8 workers sharing the GPU):
    python scripts/selfplay.py runs/gen1 --rounds 2000 --workers 8 --games-per-worker 32 \
        --sims 200 --round-seconds 120 --train-steps 500 --tb data/tb
"""
from __future__ import annotations

import argparse

from dq.model import ModelConfig
from dq.selfplay import SelfPlayConfig, run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games-per-worker", type=int, default=32)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--leaves-per-tree", type=int, default=8)
    ap.add_argument("--round-seconds", type=float, default=60.0)
    ap.add_argument("--min-buffer", type=int, default=5000)
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-sims", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--tb", default=None, help="tablebase directory (data/tb)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-autocast", action="store_true")
    ap.add_argument("--gate-revert", action="store_true", help="revert to the previous checkpoint when a gate fails")
    ap.add_argument("--small", action="store_true", help="tiny model for smoke tests")
    ap.add_argument("--tb-rescore", action="store_true",
                    help="use exact tablebase WDL as the value target for covered positions (Lc0-style rescoring)")
    ap.add_argument("--endgame-start-frac", type=float, default=0.0,
                    help="fraction of self-play games started from a random 5-8 piece endgame instead of the start position")
    ap.add_argument("--endgame-pieces", type=int, nargs=2, default=(5, 8), metavar=("MIN", "MAX"))
    ap.add_argument("--anchor", default=None, help="fixed reference checkpoint played at every gate (vs_anchor)")
    ap.add_argument("--anchor-games", type=int, default=100)
    a = ap.parse_args()
    cfg = SelfPlayConfig(workers=a.workers, games_per_worker=a.games_per_worker, simulations=a.sims,
                         leaves_per_tree=a.leaves_per_tree, round_seconds=a.round_seconds,
                         min_buffer=a.min_buffer, train_steps_per_round=a.train_steps, batch_size=a.batch,
                         lr=a.lr, gate_every=a.gate_every, eval_games=a.eval_games, eval_sims=a.eval_sims,
                         max_plies=a.max_plies, seed=a.seed, autocast=not a.no_autocast,
                         gate_revert=a.gate_revert, tb_rescore=a.tb_rescore,
                         endgame_start_frac=a.endgame_start_frac, endgame_pieces=tuple(a.endgame_pieces),
                         anchor=a.anchor, anchor_games=a.anchor_games)
    mcfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128) if a.small else ModelConfig()
    run(a.out, a.rounds, cfg, mcfg, tb_dir=a.tb, device=a.device, resume=a.resume)


if __name__ == "__main__":
    main()
