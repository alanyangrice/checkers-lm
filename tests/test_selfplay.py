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
    g.records = [(pos, {legal_moves(pos)[0]: 1.0}, 0.0), (Position.start(), {legal_moves(Position.start())[0]: 1.0}, 0.0)]
    plain, _ = _finish(g)
    assert plain[0].wdl == WDL_D and plain[1].wdl == WDL_D
    rescored, _ = _finish(g, tb, draw_rules=False)
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


def test_tb_outcome_discounts_unconvertible_wins() -> None:
    from dq.game import draw_ply_limit
    from dq.rules import Position, bit
    from dq.selfplay import tb_outcome
    from dq.tablebase import DRAW, LOSS, WIN

    class StubTB:
        def __init__(self, v, d): self.v, self.d = v, d
        def covers(self, pos): return True
        def probe(self, pos): return (self.v, self.d)

    p = Position(0, bit(18) | bit(22), 0, bit(1))          # 2 white kings vs lone black king: 7.2.8, 10 plies
    assert draw_ply_limit(p) == 10
    assert tb_outcome(StubTB(WIN, 50), p) == DRAW
    assert tb_outcome(StubTB(WIN, 50), p, draw_rules=False) == WIN
    assert tb_outcome(StubTB(LOSS, 50), p) == DRAW
    assert tb_outcome(StubTB(WIN, 8), p) == WIN
    assert tb_outcome(StubTB(WIN, 50), Position.start()) == WIN   # no counting rule applies


def test_contempt_is_root_relative_and_zero_sum() -> None:
    """v = P(w)-P(l) - c*P(d) when the leaf mover is the root side, + c*P(d) otherwise, so the
    sign-flipping backup keeps the search zero-sum (Lc0 WDL contempt)."""
    from dq.rules import BLACK, WHITE, Position, legal_moves
    from dq.model import WDL_D, WDL_L, WDL_W
    import torch

    model = _net(0)
    plain = NetEvaluator(model)
    con = NetEvaluator(model, contempt=0.3)
    p = Position.start()                       # white to move
    ms = legal_moves(p)
    (_, v0), = plain([p], [ms])
    (_, v_root), = con([p], [ms], [WHITE])     # root side == leaf mover: draws penalised
    (_, v_opp), = con([p], [ms], [BLACK])      # root side is the opponent: draws rewarded
    with torch.no_grad():
        from dq.tokens import encode_board
        wdl = torch.softmax(model(torch.tensor([encode_board(p)]))["wdl"].float(), -1)[0]
    pd = wdl[WDL_D].item()
    assert abs(v_root - (v0 - 0.3 * pd)) < 1e-5
    assert abs(v_opp - (v0 + 0.3 * pd)) < 1e-5
    (_, v_none), = con([p], [ms])              # no root info => plain value
    assert abs(v_none - v0) < 1e-6


def test_asymmetric_starts_have_strict_material_edge() -> None:
    from dq.game import GameState
    from dq.rules import legal_moves
    from dq.selfplay import SelfPlayConfig, _start_position

    tb = Tablebase(); tb.build(3)
    rng = random.Random(1)
    cfg = SelfPlayConfig(asym_start_frac=1.0, asym_weak=(3, 6), asym_extra=4)
    white_strong = 0
    for _ in range(60):
        p = _start_position(cfg, rng, tb)
        nw = bin(p.wm | p.wk).count("1"); nb = bin(p.bm | p.bk).count("1")
        assert nw != nb and 3 <= min(nw, nb) <= 6 and 1 <= abs(nw - nb) <= 4
        assert legal_moves(p) and not GameState(p).over and not tb.covers(p)
        white_strong += nw > nb
    assert 10 < white_strong < 50               # strong colour is rerolled 50/50


def test_regret_positions_and_regret_starts() -> None:
    from dq.game import GameState
    from dq.rules import Position, legal_moves
    from dq.selfplay import SelfPlayConfig, _Game, _finish, _start_position

    pos = Position.start()
    g = _Game.__new__(_Game)
    g.state = GameState(pos); g.state.result = 1.0            # white won
    mv = legal_moves(pos)[0]
    # record 1: white to move, root Q said +0.9, outcome +1 -> no regret
    # record 2: after mv, black to move, root Q said +0.8 for black, outcome -1 -> regret
    from dq.rules import apply_move
    p2 = apply_move(pos, mv)
    g.records = [(pos, {mv: 1.0}, 0.9), (p2, {legal_moves(p2)[0]: 1.0}, 0.8)]
    exs, regret = _finish(g, None, True, regret_threshold=0.5)
    assert len(exs) == 2 and regret == [p2]
    _, none = _finish(g, None, True, regret_threshold=None)
    assert none == []
    cfg = SelfPlayConfig(regret_start_frac=1.0)
    assert _start_position(cfg, random.Random(0), None, [p2]) == p2
    assert _start_position(cfg, random.Random(0), None, []) == Position.start()
