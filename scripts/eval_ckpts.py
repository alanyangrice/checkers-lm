"""Head-to-head between checkpoints, with enough games to be meaningful.

    python scripts/eval_ckpts.py runs/gen2/weights/latest.pt runs/gen1_final.pt --games 200 --sims 100 --tb data/tb

Prints the score of the first checkpoint vs each of the others (paired random openings, each
played once per colour), with a 95 % binomial interval, and the raw-policy (no search) score.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time

import torch

from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.selfplay import match
from dq.tablebase import Tablebase


def load(path: str, device) -> NetEvaluator:
    m = SquareTransformer(ModelConfig()).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return NetEvaluator(m, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--tb", default=None)
    ap.add_argument("--raw", action="store_true", help="also score A's raw policy (no search) vs B with search")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = Tablebase.load(a.tb) if a.tb else None
    ev_a = load(a.a, dev)
    for other in a.others:
        ev_b = load(other, dev)
        t = time.time()
        s = match(ev_a, ev_b, a.games, a.sims, tb, random.Random(a.seed))
        ci = 1.96 * math.sqrt(max(s * (1 - s), 1e-9) / a.games)
        w, d, l = match.last_wdl
        line = f"{a.a} vs {other}: {s:.3f} ±{ci:.3f}  W{w} D{d} L{l}  ({a.games} games, {a.sims} sims, {time.time()-t:.0f}s)"
        if a.raw:
            r = match(ev_a, ev_b, a.games // 2, a.sims, tb, random.Random(a.seed), raw_a=True)
            line += f"  raw-policy: {r:.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
