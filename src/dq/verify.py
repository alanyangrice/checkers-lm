"""Verification harness: every QA-curriculum answer and every explanation
field must be checkable programmatically against :mod:`dq.rules`.

Two layers:

1. **Question kinds** (``KINDS``).  Each kind knows how to compute the gold
   answer for a position (+ arguments), how to render it as the canonical
   text the model is trained to produce, how to parse model text back, and
   how to compare.  ``check(kind, pos, args, text)`` returns a ``Verdict``.
   Every kind takes an optional ``moves`` argument (a list of move strings)
   that is applied first; this turns the STATIC/DYNAMIC-CURRENT kinds into
   their *-FUTURE* counterparts without separate code.

2. **Explanation checks** (``check_explanation``).  Parses the structured
   tail of an explanation (BEST_MOVE / CRITICAL_LINE / PROMISING_MOVES /
   EVALUATION) and verifies legality and well-formedness.  Quality checks
   (is the best move actually good?) need the oracle and live elsewhere.

Answer text formats (canonical):
* squares: ``"5, 9, 14"`` (numeric) or ``"none"``; algebraic accepted.
* moves: full landing path, ``"22x13x6"`` / ``"21-17"``; lists comma-separated; ``"none"``.
* piece: one of ``empty, white man, white king, black man, black king``.
* counts / tempo: a bare integer.
* capture sequences: ``"all: A, B, C; maximal: A"``.
* board: a FEN ``W:W..:B..``.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from dq.rules import (
    BLACK, BK, BM, EMPTY, WHITE, WK, WM, COLOR_NAMES, Move, Position, alg_to_square, apply_move,
    capture_sequences, legal_moves, max_capture_filter, parse_move, square_to_rf, squares,
)

PIECE_NAMES = {EMPTY: "empty", WM: "white man", WK: "white king", BM: "black man", BK: "black king"}
NAME_TO_PIECE = {v: k for k, v in PIECE_NAMES.items()}


@dataclass
class Verdict:
    ok: bool
    kind: str
    expected: str
    got: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Question:
    kind: str
    args: dict[str, Any]
    prompt: str
    answer: str          # canonical gold answer text
    fen: str             # root position (before any ``moves`` prefix)


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(" .")


def parse_square_list(text: str) -> set[int]:
    t = _norm(text)
    if t in ("none", "", "no", "nothing"):
        return set()
    out = set()
    for tok in re.split(r"[,\s;]+", t):
        if not tok:
            continue
        out.add(int(tok) if tok.isdigit() else alg_to_square(tok))
    return out


def render_squares(sq: set[int] | list[int]) -> str:
    return ", ".join(str(s) for s in sorted(sq)) if sq else "none"


def parse_move_list(text: str, legal: list[Move]) -> set[Move]:
    t = _norm(text)
    if t in ("none", ""):
        return set()
    toks = [tok for tok in re.split(r"[,\s;]+", t) if tok]
    return {parse_move(tok, legal) for tok in toks}


def render_moves(moves: list[Move] | set[Move]) -> str:
    return ", ".join(str(m) for m in sorted(moves)) if moves else "none"


def apply_sequence(pos: Position, moves: list[str]) -> Position:
    for text in moves:
        pos = apply_move(pos, parse_move(text, legal_moves(pos)))
    return pos


def _colour(name: str | int) -> int:
    if isinstance(name, int):
        return name
    return {"white": WHITE, "black": BLACK, "w": WHITE, "b": BLACK}[name.lower()]


def tempo(pos: Position, colour: int) -> int:
    """Sum of advancement of a side's men (rows advanced from its own back rank)."""
    total = 0
    for s in squares(pos.men(colour)):
        r, _ = square_to_rf(s)
        total += r if colour == WHITE else 7 - r
    return total


def null_move(pos: Position) -> Position:
    return Position(pos.wm, pos.wk, pos.bm, pos.bk, pos.opponent)


def threatened(pos: Position) -> set[int]:
    """Pieces of the side to move that the opponent could capture if it were
    the opponent's turn (legal, max-capture sequences only)."""
    out: set[int] = set()
    for mv in legal_moves(null_move(pos)):
        out.update(mv.captured)
    return out


def safe_king_moves(pos: Position, square: int) -> list[Move]:
    out = []
    for mv in legal_moves(pos):
        if mv.start != square:
            continue
        after = apply_move(pos, mv)
        if not any(mv.end in reply.captured for reply in legal_moves(after)):
            out.append(mv)
    return out


# ---------------------------------------------------------------------------
# question kinds
# ---------------------------------------------------------------------------

@dataclass
class Kind:
    name: str
    stage: str                                      # STATIC-CURRENT / DYNAMIC-CURRENT
    gold: Callable[[Position, dict], Any]
    render: Callable[[Any], str]
    parse: Callable[[str, Position, dict], Any]     # text -> comparable object
    prompt: Callable[[dict], str]
    sample_args: Callable[[Position, random.Random], dict | None]


def _piece_on_square(pos, a):
    return pos.piece_at(a["square"])


def _locate(pos, a):
    c = _colour(a["colour"])
    return set(squares(pos.kings(c) if a.get("piece", "king") == "king" else pos.men(c)))


def _count(pos, a):
    c = _colour(a["colour"])
    kind = a.get("piece", "all")
    m = {"man": pos.men(c), "king": pos.kings(c), "all": pos.pieces(c)}[kind]
    return bin(m).count("1")


def _legal_for_piece(pos, a):
    return {m for m in legal_moves(pos) if m.start == a["square"]}


def _all_legal(pos, a):
    return set(legal_moves(pos))


def _captures(pos, a):
    seqs = capture_sequences(pos)
    return (set(seqs), set(max_capture_filter(seqs)))


def _render_captures(v):
    allseq, maxseq = v
    return f"all: {render_moves(allseq)}; maximal: {render_moves(maxseq)}"


def _parse_captures(text, pos, a):
    t = _norm(text)
    m = re.match(r"all:\s*(.*?);\s*maximal:\s*(.*)$", t)
    if not m:
        raise ValueError("expected 'all: ...; maximal: ...'")
    seqs = capture_sequences(pos)
    return (parse_move_list(m.group(1), seqs), parse_move_list(m.group(2), seqs))


def _sample_square_any(pos, rng):
    return {"square": rng.randint(1, 32)}


def _sample_own_piece(pos, rng):
    own = squares(pos.pieces(pos.turn))
    return {"square": rng.choice(own)} if own else None


def _sample_own_king(pos, rng):
    ks = squares(pos.kings(pos.turn))
    return {"square": rng.choice(ks)} if ks else None


def _sample_colour(pos, rng):
    return {"colour": COLOR_NAMES[rng.randrange(2)]}


def _sample_colour_piece(pos, rng):
    return {"colour": COLOR_NAMES[rng.randrange(2)], "piece": rng.choice(["man", "king", "all"])}


KINDS: dict[str, Kind] = {}


def _register(k: Kind) -> None:
    KINDS[k.name] = k


_register(Kind("piece_on_square", "STATIC-CURRENT", _piece_on_square, lambda v: PIECE_NAMES[v],
               lambda t, p, a: NAME_TO_PIECE[_norm(t)],
               lambda a: f"What is on square {a['square']}?", _sample_square_any))
_register(Kind("locate_kings", "STATIC-CURRENT", _locate, render_squares,
               lambda t, p, a: parse_square_list(t),
               lambda a: f"On which squares are the {a['colour']} kings?", _sample_colour))
_register(Kind("count", "STATIC-CURRENT", _count, str, lambda t, p, a: int(_norm(t)),
               lambda a: f"How many {a['colour']} {'pieces' if a.get('piece', 'all') == 'all' else a['piece'] + 's'} are there?",
               _sample_colour_piece))
_register(Kind("tempo", "STATIC-CURRENT", lambda p, a: tempo(p, _colour(a["colour"])), str,
               lambda t, p, a: int(_norm(t)),
               lambda a: f"What is the tempo count (total advancement of men) for {a['colour']}?", _sample_colour))
_register(Kind("legal_moves_for_piece", "DYNAMIC-CURRENT", _legal_for_piece, render_moves,
               lambda t, p, a: parse_move_list(t, legal_moves(p)),
               lambda a: f"List the legal moves for the piece on square {a['square']}.", _sample_own_piece))
_register(Kind("legal_moves", "DYNAMIC-CURRENT", _all_legal, render_moves,
               lambda t, p, a: parse_move_list(t, legal_moves(p)),
               lambda a: "List all legal moves for the side to move.", lambda p, r: {}))
_register(Kind("capture_sequences", "DYNAMIC-CURRENT", _captures, _render_captures, _parse_captures,
               lambda a: "List every complete capture sequence available to the side to move, and which of them are maximal (legal).",
               lambda p, r: {}))
_register(Kind("threatened", "DYNAMIC-CURRENT", lambda p, a: threatened(p), render_squares,
               lambda t, p, a: parse_square_list(t),
               lambda a: "Which pieces of the side to move could the opponent capture if it were the opponent's turn?",
               lambda p, r: {}))
_register(Kind("safe_king_moves", "DYNAMIC-CURRENT", lambda p, a: set(safe_king_moves(p, a["square"])), render_moves,
               lambda t, p, a: parse_move_list(t, legal_moves(p)),
               lambda a: f"Which moves of the king on square {a['square']} leave it safe from immediate capture?",
               _sample_own_king))
_register(Kind("board", "STATIC-CURRENT", lambda p, a: p, lambda v: v.fen(),
               lambda t, p, a: Position.from_fen(t.strip()),
               lambda a: "Give the position as a FEN string.", lambda p, r: {}))

STAGES = {
    "STATIC-CURRENT": [k for k, v in KINDS.items() if v.stage == "STATIC-CURRENT"],
    "DYNAMIC-CURRENT": [k for k, v in KINDS.items() if v.stage == "DYNAMIC-CURRENT"],
}


def target_position(pos: Position, args: dict) -> Position:
    return apply_sequence(pos, list(args.get("moves", ())))


def gold_answer(kind: str, pos: Position, args: dict) -> str:
    k = KINDS[kind]
    return k.render(k.gold(target_position(pos, args), args))


def check(kind: str, pos: Position, args: dict, text: str) -> Verdict:
    """Verify model ``text`` against the gold answer for ``kind`` at ``pos``."""
    k = KINDS[kind]
    try:
        tgt = target_position(pos, args)
    except ValueError as e:
        return Verdict(False, kind, "", text, f"bad move prefix: {e}")
    gold = k.gold(tgt, args)
    expected = k.render(gold)
    try:
        got = k.parse(text, tgt, args)
    except (ValueError, KeyError, IndexError) as e:
        return Verdict(False, kind, expected, text, f"unparseable: {e}")
    return Verdict(got == gold, kind, expected, text, "" if got == gold else "mismatch")


def make_question(kind: str, pos: Position, rng: random.Random, moves: list[str] | None = None) -> Question | None:
    """Sample a question of ``kind`` about ``pos`` (after the optional move
    prefix).  Returns None if the kind does not apply (e.g. no kings)."""
    k = KINDS[kind]
    tgt = apply_sequence(pos, moves or [])
    args = k.sample_args(tgt, rng)
    if args is None:
        return None
    if moves:
        args = {**args, "moves": list(moves)}
    prompt = k.prompt(args)
    if moves:
        prompt = f"After the moves {' '.join(moves)}: {prompt}"
    return Question(kind, args, prompt, k.render(k.gold(tgt, args)), pos.fen())


def random_line(pos: Position, rng: random.Random, length: int) -> list[str]:
    """A random legal move sequence of up to ``length`` plies (stops early at
    game end)."""
    out = []
    for _ in range(length):
        lm = legal_moves(pos)
        if not lm:
            break
        mv = rng.choice(lm)
        out.append(str(mv))
        pos = apply_move(pos, mv)
    return out


# ---------------------------------------------------------------------------
# explanation structure
# ---------------------------------------------------------------------------

FIELD_RE = re.compile(r"^\s*(BEST_MOVE|CRITICAL_LINE|PROMISING_MOVES|EVALUATION)\s*:\s*(.*?)\s*$", re.M)
WDL_RE = re.compile(r"W\s*(\d+(?:\.\d+)?)\s*%?\s*D\s*(\d+(?:\.\d+)?)\s*%?\s*L\s*(\d+(?:\.\d+)?)\s*%?", re.I)


@dataclass
class Explanation:
    prose: str
    best_move: str | None = None
    critical_line: list[str] = field(default_factory=list)
    promising_moves: list[str] = field(default_factory=list)
    evaluation: tuple[float, float, float] | None = None


def parse_explanation(text: str) -> Explanation:
    fields = {m.group(1): m.group(2) for m in FIELD_RE.finditer(text)}
    first = FIELD_RE.search(text)
    ex = Explanation(prose=text[: first.start()].strip() if first else text.strip())
    ex.best_move = fields.get("BEST_MOVE") or None
    ex.critical_line = [t for t in re.split(r"[\s,]+", fields.get("CRITICAL_LINE", "")) if t]
    ex.promising_moves = [t for t in re.split(r"[\s,;]+", fields.get("PROMISING_MOVES", "")) if t]
    ev = fields.get("EVALUATION")
    if ev:
        m = WDL_RE.search(ev)
        if m:
            ex.evaluation = tuple(float(x) for x in m.groups())  # type: ignore[assignment]
    return ex


def check_explanation(pos: Position, text: str, max_promising: int = 3) -> list[Verdict]:
    """Structural checks on an explanation: all moves legal, line playable,
    promising moves distinct and include the best move, WDL sums to 100."""
    ex = parse_explanation(text)
    out: list[Verdict] = []
    lm = legal_moves(pos)
    best: Move | None = None

    if ex.best_move is None:
        out.append(Verdict(False, "best_move", "a legal move", "missing", "BEST_MOVE absent"))
    else:
        try:
            best = parse_move(ex.best_move, lm)
            out.append(Verdict(True, "best_move", str(best), ex.best_move))
        except ValueError as e:
            out.append(Verdict(False, "best_move", render_moves(lm), ex.best_move, str(e)))

    if not ex.critical_line:
        out.append(Verdict(False, "critical_line", "a legal line", "missing", "CRITICAL_LINE absent"))
    else:
        p, ok, reason, played = pos, True, "", []
        for i, tok in enumerate(ex.critical_line):
            try:
                mv = parse_move(tok, legal_moves(p))
            except ValueError as e:
                ok, reason = False, f"ply {i + 1} ({tok}): {e}"
                break
            played.append(mv)
            p = apply_move(p, mv)
        if ok and best is not None and played and played[0] != best:
            ok, reason = False, "line does not start with BEST_MOVE"
        out.append(Verdict(ok, "critical_line", "legal line starting with BEST_MOVE", " ".join(ex.critical_line), reason))

    if not ex.promising_moves:
        out.append(Verdict(False, "promising_moves", "1-3 legal moves", "missing", "PROMISING_MOVES absent"))
    else:
        parsed: list[Move] = []
        reason = ""
        for tok in ex.promising_moves:
            try:
                parsed.append(parse_move(tok, lm))
            except ValueError as e:
                reason = str(e)
                break
        if not reason and len(set(parsed)) != len(parsed):
            reason = "duplicate promising moves"
        if not reason and len(parsed) > max_promising:
            reason = f"more than {max_promising} promising moves"
        if not reason and best is not None and best not in parsed:
            reason = "BEST_MOVE not among PROMISING_MOVES"
        out.append(Verdict(not reason, "promising_moves", "distinct legal moves incl. best", ", ".join(ex.promising_moves), reason))

    if ex.evaluation is None:
        out.append(Verdict(False, "evaluation", "W d% D d% L d%", "missing/unparseable", "EVALUATION absent"))
    else:
        total = sum(ex.evaluation)
        ok = abs(total - 100.0) <= 1.5 and all(0 <= x <= 100 for x in ex.evaluation)
        out.append(Verdict(ok, "evaluation", "sums to 100", f"W {ex.evaluation[0]} D {ex.evaluation[1]} L {ex.evaluation[2]}",
                           "" if ok else f"sums to {total}"))
    return out


__all__ = [
    "Verdict", "Question", "KINDS", "STAGES", "check", "gold_answer", "make_question", "random_line",
    "apply_sequence", "tempo", "threatened", "safe_king_moves", "parse_explanation", "check_explanation",
    "render_moves", "render_squares", "parse_move_list", "parse_square_list",
]
