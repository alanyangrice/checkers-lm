"""Cross-check the native generator against pydraughts (the reference)."""
import random

import pytest

from dq.ref_pydraughts import RefBoard, ref_legal_moves
from dq.rules import Position, apply_move, legal_moves, random_position


def _same(pos: Position) -> None:
    mine = sorted(legal_moves(pos))
    ref = sorted(ref_legal_moves(pos))
    assert mine == ref, f"\n{pos.fen()}\n{pos.ascii()}\nmine={list(map(str, mine))}\nref ={list(map(str, ref))}"


@pytest.mark.parametrize("seed", range(40))
def test_random_playout_matches(seed: int) -> None:
    rng = random.Random(seed)
    pos = Position.start()
    ref = RefBoard(pos)
    for ply in range(160):
        mine = sorted(legal_moves(pos))
        theirs = sorted(ref.legal_moves())
        assert mine == theirs, f"ply {ply} {pos.fen()}: {list(map(str, mine))} vs {list(map(str, theirs))}"
        if not mine:
            break
        mv = rng.choice(mine)
        pos = apply_move(pos, mv)
        ref.push(mv)
        assert ref.position() == pos, f"after {mv} from ply {ply}"


@pytest.mark.parametrize("seed", range(8))
def test_random_positions_match(seed: int) -> None:
    rng = random.Random(1000 + seed)
    for _ in range(60):
        pos = random_position(rng, rng.randint(1, 8), rng.randint(1, 8), king_prob=0.4)
        _same(pos)


def test_known_tricky_positions_match() -> None:
    fens = [
        "W:WK1:B10",                 # flying king landing choices
        "W:W11:B7,6",                # back-rank pass-through, no promotion
        "W:W11,22:B15,23,17",        # max-capture rule
        "W:WK28:B24,15,6",           # king multi-capture along a diagonal with gaps
        "B:W21,22,23:BK4,12",        # black king vs men
        "W:WK14:B10,11,17,18,19,9",  # king surrounded: many branching sequences
    ]
    for f in fens:
        _same(Position.from_fen(f))
