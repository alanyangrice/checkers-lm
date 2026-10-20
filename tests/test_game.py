import random
from collections import Counter

from dq.game import GameState, active_draw_rule, play_game, random_policy
from dq.rules import Position, legal_moves, parse_move


def test_random_games_terminate() -> None:
    rng = random.Random(0)
    reasons: Counter[str] = Counter()
    for _ in range(40):
        g = play_game(random_policy(rng), random_policy(rng), max_plies=600)
        assert g.over and g.result in (0.0, 0.5, 1.0)
        reasons[g.reason] += 1
    assert reasons["no moves"] > 0
    assert sum(reasons.values()) == 40


def test_threefold() -> None:
    # two kings shuffling: W Ka1<->b2, B Kh8<->g7 ... but a1/h8 corner on the
    # long diagonal is a capture; use kings far apart off the diagonal.
    g = GameState(Position.from_fen("W:WK31:BK2"))     # f1 vs d8
    # rule 7.2.6? total 2 pieces, both kings -> 7.2.5 (30 king moves); threefold comes first
    seq = ["31-27", "2-6", "27-31", "6-2"] * 2 + ["31-27", "2-6", "27-31", "6-2"]
    for s in seq:
        if g.over:
            break
        g.push(parse_move(s, g.legal()))
    assert g.over and g.result == 0.5 and g.reason == "7.2.3 threefold"


def test_rule_728_two_kings_vs_one() -> None:
    pos = Position.from_fen("W:WK31,K27:BK2")
    r = active_draw_rule(pos, (0, 0))
    assert r is not None and r.article == "7.2.8" and r.limit == 10


def test_rule_724_three_kings_vs_one() -> None:
    pos = Position.from_fen("W:WK31,K27,K23:BK2")
    r = active_draw_rule(pos, (0, 0))
    assert r is not None and r.article == "7.2.4" and r.limit == 30


def test_rule_726_material_buckets() -> None:
    assert active_draw_rule(Position.from_fen("W:WK31,27:BK2,6"), (0, 0)).limit == 60    # 4 pieces
    assert active_draw_rule(Position.from_fen("W:WK31,27,26:BK2,6,7"), (0, 0)).limit == 120  # 6
    assert active_draw_rule(Position.from_fen("W:WK31,27,26,25:BK2,6,7,8"), (0, 0)).article == "7.2.5"
    assert active_draw_rule(Position.from_fen("W:W27,26:B6,7"), (0, 0)) is None  # no kings


def test_counter_resets_on_capture_and_rule_switches() -> None:
    g = GameState(Position.from_fen("W:WK29,K31:BK2,K22"))   # 4 pieces -> 7.2.6 (60)
    assert g.rule.article == "7.2.6"
    mv = next(m for m in g.legal() if m.is_capture)           # a1 x c3
    g.push(mv)
    assert g.draw_counter == 0
    assert g.rule.article == "7.2.8" and g.rule.limit == 10   # 2 vs lone king


def test_white_wins_when_black_has_no_moves() -> None:
    # Black man on h8 (4) can only go to g7 (8); 8 is held by a white king
    # that cannot be captured because 11 is occupied too.
    g = GameState(Position.from_fen("W:WK8,K11,K31:B4"))
    g.push(parse_move("31-27", g.legal()))
    assert g.over and g.result == 1.0 and g.reason == "no moves"
