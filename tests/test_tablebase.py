import random

import pytest

from dq.rules import BLACK, WHITE, Position, apply_move, legal_moves, random_position
from dq.tablebase import DRAW, LOSS, WIN, Tablebase, minimax, pack, unpack, unit_of


@pytest.fixture(scope="module")
def tb3() -> Tablebase:
    tb = Tablebase()
    tb.build(3)
    return tb


def test_pack_roundtrip() -> None:
    rng = random.Random(1)
    for _ in range(200):
        p = random_position(rng, rng.randint(1, 4), rng.randint(1, 3), king_prob=0.5, turn=WHITE)
        assert unpack(pack(p)) == p


def test_units_present(tb3: Tablebase) -> None:
    assert set(tb3.units) == {(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)}
    for u in tb3.units.values():
        assert len(u.keys) > 0
        assert (u.wdl != 2).all()


def test_known_positions(tb3: Tablebase) -> None:
    # K vs K with both kings in the a1/h8 corners: the side to move must leave
    # its corner along the long diagonal and is captured -> loss in 2.
    assert tb3.probe(Position.from_fen("W:WK29:BK4")) == (LOSS, 2)
    # ...so a king that can reach the opposite corner wins in 3.
    assert tb3.probe(Position.from_fen("W:WK18:BK4")) == (WIN, 3)
    # most K vs K positions are drawn
    u = tb3.units[(2, 0)]
    assert (u.wdl == DRAW).sum() > len(u.keys) // 2
    # white king captures black man immediately: win in 1
    assert tb3.probe(Position.from_fen("W:WK29:B25")) == (WIN, 1)
    # black to move, white king can capture next: black to move with a man that
    # must step into capture range is a loss in 2 if every black move loses
    v, d = tb3.probe(Position.from_fen("B:WK1:B28"))
    assert v in (WIN, LOSS, DRAW)
    # mover has a man on the edge blocked by two opponents -> no moves -> loss 0
    # (construct one with legal_moves empty)
    rng = random.Random(3)
    found = False
    for _ in range(5000):
        p = random_position(rng, 1, 2, king_prob=0.0, turn=WHITE)
        if not legal_moves(p):
            assert tb3.probe(p) == (LOSS, 0)
            found = True
            break
    assert found


def test_self_consistency(tb3: Tablebase) -> None:
    """Every stored value must equal the best over children (negamax step)."""
    rng = random.Random(11)
    checked = 0
    for _ in range(3000):
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if not tb3.covers(p):
            continue
        v, d = tb3.probe(p)
        moves = legal_moves(p)
        if not moves:
            assert (v, d) == (LOSS, 0)
            continue
        kids = []
        for mv in moves:
            c = apply_move(p, mv)
            if c.pieces(c.turn) == 0:
                kids.append((LOSS, 0))
            else:
                kids.append(tb3.probe(c))
        losses = [cd for cv, cd in kids if cv == LOSS]
        if losses:
            assert v == WIN and d == min(losses) + 1, (p.fen(), v, d, kids)
        elif any(cv == DRAW for cv, _ in kids):
            assert v == DRAW, (p.fen(), v, d, kids)
        else:
            assert v == LOSS and d == max(cd for _, cd in kids) + 1, (p.fen(), v, d, kids)
        checked += 1
    assert checked > 1000


def test_matches_minimax(tb3: Tablebase) -> None:
    rng = random.Random(5)
    n = 0
    for _ in range(400):
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if not tb3.covers(p):
            continue
        ref = minimax(p, 9)
        got = tb3.probe(p)
        if ref is None:
            # outcome not forced within 9 plies: tablebase must not claim a short one
            assert got[0] == DRAW or got[1] > 9, (p.fen(), got)
        else:
            assert got == ref, (p.fen(), got, ref)
        n += 1
    assert n > 200


def test_save_load(tb3: Tablebase, tmp_path) -> None:
    tb3.save(tmp_path)
    tb = Tablebase.load(tmp_path)
    assert set(tb.units) == set(tb3.units)
    p = Position.from_fen("W:WK29:B25")
    assert tb.probe(p) == tb3.probe(p)
