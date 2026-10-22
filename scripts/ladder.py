"""Round-robin between checkpoints and a Bradley-Terry / Elo fit.

    python scripts/ladder.py runs/random_init.pt runs/gen1_final.pt ... --games 100 --sims 100 --tb data/tb

Every pair plays `games` games (paired random openings, each colour once) at `sims` nodes with
tablebases; Elo is fitted by maximum likelihood with draws counted as half points and the first
checkpoint pinned at 0.  Prints the pairwise table and the ladder.  Note these are *internal*
ratings: only differences between the listed models are meaningful.
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import time

import torch

from dq.model import NetEvaluator, load_model
from dq.selfplay import match
from dq.tablebase import Tablebase


def fit_elo(names, scores, games):
    """scores[(a,b)] = A's score fraction vs B over games[(a,b)] games.  Gradient ascent on the
    Bradley-Terry log-likelihood with draws as half points; model 0 pinned at 0 Elo."""
    r = {n: 0.0 for n in names}
    for _ in range(5000):
        grad = {n: 0.0 for n in names}
        for (a, b), s in scores.items():
            n = games[(a, b)]
            p = 1 / (1 + 10 ** ((r[b] - r[a]) / 400))
            grad[a] += n * (s - p); grad[b] -= n * (s - p)
        for n in names[1:]:
            r[n] += 4.0 * grad[n] / max(sum(games[k] for k in games if n in k), 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--tb", default="data/tb")
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = Tablebase.load(a.tb)
    evs = {}
    for c in a.ckpts:
        m = load_model(c, dev); evs[c] = NetEvaluator(m, dev)
        print(f"{c}: {m.n_params():,} params", flush=True)
    scores, games = {}, {}
    for x, y in itertools.combinations(a.ckpts, 2):
        t = time.time()
        s = match(evs[x], evs[y], a.games, a.sims, tb, random.Random(a.seed))
        w, d, l = match.last_wdl
        scores[(x, y)] = s; games[(x, y)] = a.games
        print(f"{x} vs {y}: {s:.3f}  W{w} D{d} L{l}  ({time.time()-t:.0f}s)", flush=True)
    r = fit_elo(a.ckpts, scores, games)
    print("\nInternal Elo ladder (first model pinned at 0):")
    for n in sorted(a.ckpts, key=lambda k: -r[k]):
        print(f"  {r[n]:+7.0f}  {n}")


if __name__ == "__main__":
    main()
