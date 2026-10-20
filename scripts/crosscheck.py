"""Long-running cross-check of dq.rules against pydraughts.

    .venv/bin/python scripts/crosscheck.py --games 300 --positions 5000 --seed 0

Prints the first disagreement (FEN, both move lists) and exits non-zero.
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from dq.ref_pydraughts import RefBoard, ref_legal_moves
from dq.rules import Position, apply_move, legal_moves, random_position


def compare(pos: Position, where: str) -> bool:
    mine = sorted(legal_moves(pos))
    ref = sorted(ref_legal_moves(pos))
    if mine != ref:
        print(f"MISMATCH at {where}\n{pos.fen()}\n{pos.ascii()}")
        print("native :", [str(m) for m in mine])
        print("pydraughts:", [str(m) for m in ref])
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--positions", type=int, default=3000)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    checked = 0
    for g in range(a.games):
        pos = Position.start()
        ref = RefBoard(pos)
        for ply in range(a.max_plies):
            mine = sorted(legal_moves(pos))
            theirs = sorted(ref.legal_moves())
            checked += 1
            if mine != theirs:
                compare(pos, f"game {g} ply {ply}")
                return 1
            if not mine:
                break
            mv = rng.choice(mine)
            pos = apply_move(pos, mv)
            ref.push(mv)
            if ref.position() != pos:
                print(f"STATE MISMATCH game {g} ply {ply} after {mv}: {pos.fen()} vs {ref.position().fen()}")
                return 1
    for i in range(a.positions):
        pos = random_position(rng, rng.randint(1, 12), rng.randint(1, 12), king_prob=rng.choice([0.1, 0.3, 0.6]))
        checked += 1
        if not compare(pos, f"random position {i}"):
            return 1
    print(f"OK: {checked} positions agree ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
