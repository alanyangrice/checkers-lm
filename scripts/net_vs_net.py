"""How much of a checkpoint's edge is the *network* rather than the search + tablebases?

    python scripts/net_vs_net.py A.pt B.pt --tb data/tb

Plays A vs B as raw policies (1-node, no search), and with search at several node counts.
Search and tablebases are shared by both sides, so differences shrink as sims grow.
"""
from __future__ import annotations

import argparse
import random

import torch

from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.selfplay import match
from dq.tablebase import Tablebase


def load(path, dev):
    from dq.model import load_model
    return NetEvaluator(load_model(path, dev), dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--tb", default="data/tb")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, nargs="+", default=[10, 100, 400])
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = Tablebase.load(a.tb)
    ea, eb = load(a.a, dev), load(a.b, dev)
    s = match(ea, eb, a.games, 1, tb, random.Random(a.seed), raw_a=True)
    w, d, l = match.last_wdl
    print(f"raw policy A vs 1-node B (net vs net):   {s:.3f}  W{w} D{d} L{l}", flush=True)
    for sims in a.sims:
        s = match(ea, eb, a.games, sims, tb, random.Random(a.seed))
        w, d, l = match.last_wdl
        print(f"search {sims:>4} sims, both sides:          {s:.3f}  W{w} D{d} L{l}", flush=True)


if __name__ == "__main__":
    main()
