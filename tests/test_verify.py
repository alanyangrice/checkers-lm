import random

from dq.rules import BLACK, WHITE, Position, apply_move, legal_moves, random_position
from dq.verify import (
    KINDS, STAGES, check, check_explanation, gold_answer, make_question, random_line, safe_king_moves, tempo,
    threatened,
)


def test_static_kinds_on_start_position():
    p = Position.start()
    assert check("piece_on_square", p, {"square": 1}, "black man")
    assert check("piece_on_square", p, {"square": 32}, "White Man ")
    assert not check("piece_on_square", p, {"square": 32}, "black man")
    assert not check("piece_on_square", p, {"square": 15}, "white man")
    assert check("piece_on_square", p, {"square": 15}, "empty")
    assert check("locate_kings", p, {"colour": "white"}, "none")
    assert check("count", p, {"colour": "black", "piece": "man"}, "12")
    assert check("count", p, {"colour": "black", "piece": "king"}, "0")
    assert not check("count", p, {"colour": "black"}, "eleven")
    assert tempo(p, WHITE) == 12 * 1 and tempo(p, BLACK) == 12  # three rows of four men at advancement 0,1,2
    assert check("tempo", p, {"colour": "white"}, "12")
    assert check("board", p, {}, p.fen())


def test_dynamic_kinds():
    p = Position.from_fen("W:W11,22:B15,23,17")
    assert gold_answer("capture_sequences", p, {}) == "all: 11x18x27, 22x13; maximal: 11x18x27"
    assert check("capture_sequences", p, {}, "all: 22x13, 11x18x27; maximal: 11x18x27")
    assert check("capture_sequences", p, {}, "all: 22x13, f6xd4xf2; maximal: 11x27")   # short/algebraic forms
    assert not check("capture_sequences", p, {}, "all: 11x18x27; maximal: 11x18x27")
    assert check("legal_moves_for_piece", p, {"square": 22}, "none")
    assert check("legal_moves_for_piece", p, {"square": 11}, "11x18x27")
    assert check("legal_moves", p, {}, "11x18x27")
    # threatened: black to move would capture? swap turn
    q = Position.from_fen("B:W11,22:B15,23,17")
    assert threatened(q) == {15, 23}          # white's max capture takes both
    assert check("threatened", q, {}, "23, 15")
    assert not check("threatened", q, {}, "15")


def test_safe_king_moves():
    # white king on 1 (b8), black men on 10 (d6) guarded: after 1x15 the king on e5 is recaptured by black 20? set up
    p = Position.from_fen("W:WK1:B10,24")       # black d6 and g3
    # captures: 1x15, 1x19 (then g3 takes f4? g3 x f4 -> e5 needs 15 empty: yes) ...
    safe = {str(m) for m in safe_king_moves(p, 1)}
    all_moves = {str(m) for m in legal_moves(p) if m.start == 1}
    assert safe <= all_moves
    for m in legal_moves(p):
        after = apply_move(p, m)
        attacked = any(m.end in r.captured for r in legal_moves(after))
        assert (str(m) in safe) == (not attacked)


def test_future_kinds_via_move_prefix():
    p = Position.start()
    args = {"moves": ["23-19", "10-14"], "square": 19}
    tgt = Position.from_fen(gold_answer("board", p, {"moves": args["moves"]}))
    assert tgt.turn == WHITE and tgt.piece_at(19) != 0
    assert check("legal_moves_for_piece", p, args, gold_answer("legal_moves_for_piece", p, args))
    assert not check("legal_moves_for_piece", p, {"moves": ["23-19", "10-14", "99-1"], "square": 19}, "none")


def test_generated_questions_self_consistent():
    rng = random.Random(42)
    n = 0
    for _ in range(150):
        p = random_position(rng, rng.randint(1, 10), rng.randint(1, 10), king_prob=0.3)
        if not legal_moves(p):
            continue
        line = random_line(p, rng, rng.randint(0, 6)) if rng.random() < 0.5 else None
        for kind in KINDS:
            q = make_question(kind, p, rng, line)
            if q is None:
                continue
            v = check(kind, p, q.args, q.answer)
            assert v, (kind, q, v)
            n += 1
    assert n > 500
    assert set(STAGES["STATIC-CURRENT"]) | set(STAGES["DYNAMIC-CURRENT"]) == set(KINDS)


def test_explanation_checks():
    p = Position.start()
    text = (
        "White should develop the centre.\n"
        "BEST_MOVE: 23-19\nCRITICAL_LINE: 23-19 10-14 22-18\nPROMISING_MOVES: 23-19, 22-18, 24-20\n"
        "EVALUATION: W 48% D 40% L 12%"
    )
    vs = check_explanation(p, text)
    assert all(vs), [v for v in vs if not v]
    bad = text.replace("CRITICAL_LINE: 23-19 10-14 22-18", "CRITICAL_LINE: 23-19 10-14 22-19")
    vs = {v.kind: v for v in check_explanation(p, bad)}
    assert not vs["critical_line"] and vs["best_move"]
    bad = text.replace("EVALUATION: W 48% D 40% L 12%", "EVALUATION: W 70% D 40% L 12%")
    assert not {v.kind: v for v in check_explanation(p, bad)}["evaluation"]
    bad = text.replace("PROMISING_MOVES: 23-19, 22-18, 24-20", "PROMISING_MOVES: 22-18, 24-20")
    assert not {v.kind: v for v in check_explanation(p, bad)}["promising_moves"]
    bad = text.replace("BEST_MOVE: 23-19", "BEST_MOVE: 23-20")
    assert not {v.kind: v for v in check_explanation(p, bad)}["best_move"]
