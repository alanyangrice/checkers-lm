import random

import pytest

torch = pytest.importorskip("torch")

from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.selfplay import match, tablebase_wdl_error
from dq.tablebase import Tablebase


def _net(seed: int) -> SquareTransformer:
    torch.manual_seed(seed)
    return SquareTransformer(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)).eval()


def test_batched_match_is_symmetric_and_handles_last_capture() -> None:
    """Games that end mid-round (a capture of the last piece) must not be
    searched again; A-vs-B and B-vs-A must mirror (search is deterministic
    at T=0 without noise)."""
    tb = Tablebase(); tb.build(3)
    a, b = NetEvaluator(_net(1)), NetEvaluator(_net(2))
    ab = match(a, b, games=20, sims=16, tb=tb, rng=random.Random(0), max_plies=300)
    ba = match(b, a, games=20, sims=16, tb=tb, rng=random.Random(0), max_plies=300)
    assert abs(ab + ba - 1.0) < 1e-9
    aa = match(a, a, games=10, sims=8, tb=None, rng=random.Random(0), max_plies=300)
    assert aa == 0.5   # paired openings + deterministic search => exact mirror
    s_raw = match(a, b, games=4, sims=8, tb=None, rng=random.Random(0), max_plies=120, raw_a=True)
    assert 0.0 <= s_raw <= 1.0


def test_tablebase_wdl_error_in_range() -> None:
    tb = Tablebase(); tb.build(2)
    e = tablebase_wdl_error(_net(3), tb, 100, random.Random(1), torch.device("cpu"))
    assert 0.0 <= e <= 1.0


def test_tablebase_rescoring_overrides_game_outcome() -> None:
    """A covered position gets the exact tablebase WDL as value target, regardless of how
    the game actually ended; uncovered positions keep the game outcome."""
    from dq.model import WDL_D, WDL_L, WDL_W
    from dq.rules import WHITE, Position, legal_moves, random_position
    from dq.selfplay import _Game, _finish
    from dq.tablebase import DRAW, LOSS, WIN

    tb = Tablebase(); tb.build(3)
    rng = random.Random(3)
    # a decisive 3-piece position (side to move wins or loses by tablebase)
    pos = None
    while pos is None:
        cand = random_position(rng, 2, 1)
        r = tb.probe(cand)
        if r is not None and r[0] != DRAW and legal_moves(cand):
            pos = cand
    g = _Game.__new__(_Game)
    from dq.game import GameState
    g.state = GameState(pos); g.state.result = 0.5          # pretend the game was drawn
    g.records = [(pos, {legal_moves(pos)[0]: 1.0}), (Position.start(), {legal_moves(Position.start())[0]: 1.0})]
    plain = _finish(g)
    assert plain[0].wdl == WDL_D and plain[1].wdl == WDL_D
    rescored = _finish(g, tb)
    expect = WDL_W if tb.probe(pos)[0] == WIN else WDL_L
    assert rescored[0].wdl == expect           # covered: exact tablebase value
    assert rescored[1].wdl == WDL_D            # start position: not covered, keeps outcome


def test_endgame_start_positions_are_legal_and_uncovered() -> None:
    from dq.game import GameState
    from dq.rules import Position, legal_moves
    from dq.selfplay import SelfPlayConfig, _start_position

    tb = Tablebase(); tb.build(3)
    rng = random.Random(0)
    cfg = SelfPlayConfig(endgame_start_frac=1.0, endgame_pieces=(4, 6))
    seen_nonstart = 0
    for _ in range(50):
        p = _start_position(cfg, rng, tb)
        assert legal_moves(p) and not GameState(p).over
        assert not tb.covers(p)
        n = bin(p.wm | p.wk | p.bm | p.bk).count("1")
        assert 4 <= n <= 6
        seen_nonstart += p != Position.start()
    assert seen_nonstart == 50
    cfg0 = SelfPlayConfig(endgame_start_frac=0.0)
    assert all(_start_position(cfg0, rng, tb) == Position.start() for _ in range(10))
