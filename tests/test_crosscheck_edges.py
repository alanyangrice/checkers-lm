"""Targeted edge-case cross-check against pydraughts (short version of
scripts/crosscheck_edges.py).  See that script for the shapes covered."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from crosscheck_edges import GENS, _swap  # noqa: E402

from dq.ref_pydraughts import ref_legal_moves
from dq.rules import legal_moves


@pytest.mark.parametrize("kind", sorted(GENS))
def test_edge_kind_matches(kind: str) -> None:
    rng = random.Random(hash(kind) & 0xFFFF)
    gen = GENS[kind]
    n = 0
    tries = 0
    while n < 150 and tries < 10_000:
        tries += 1
        pos = gen(rng)
        if pos is None:
            continue
        for p in (pos, _swap(pos)):
            try:
                p.validate()
            except ValueError:
                continue
            mine = sorted(legal_moves(p))
            ref = sorted(ref_legal_moves(p))
            assert mine == ref, (
                f"kind={kind}\n{p.fen()}\n{p.ascii()}\n"
                f"native={list(map(str, mine))}\nref   ={list(map(str, ref))}"
            )
            n += 1
    assert n >= 150, f"generator {kind} produced only {n} positions"
