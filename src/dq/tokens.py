"""Token encoding for the expert encoder and the LM bridge (PLAN §4).

Two vocabularies:

* **Board tokens** — what the encoder consumes.  Always from the point of
  view of the side to move (``Position.pov()``), so the mover is "me",
  moves toward square 1, and the model never needs a side-to-move flag.
  One token per playable square, in square order 1..32::

      EMPTY_TOK=0  MM=1 (my man)  MK=2 (my king)  OM=3 (opp man)  OK=4 (opp king)

  ``encode_board(pos)`` -> ``list[int]`` of length 32.
  ``decode_board(ids, turn)`` -> ``Position`` (inverse, given the real turn).

* **Text tokens** — the special strings added to the LM tokenizer so that
  squares and pieces are single tokens with no linguistic prior:

      <SQ_1> .. <SQ_32>          squares, POV-relative (after rotation)
      <PIECE_MM> <PIECE_MK> <PIECE_OM> <PIECE_OK>
      <WHITE> <BLACK>            absolute colours, for consolidated prose

  ``move_to_text(mv)`` renders a (POV) move as ``<SQ_a><SQ_b>`` or
  ``<SQ_a>x<SQ_b>x<SQ_c>``; ``text_to_move`` parses it back against a legal
  list.  ``pov_move`` / ``unpov_move`` convert between absolute and POV
  square numbering (``s -> 33 - s`` when Black is to move).

Keep these pure and dependency-free; the tokenizer registration lives in
the training code.
"""
from __future__ import annotations

import re
from typing import Iterable

from dq.rules import BLACK, WHITE, Move, Position, bit, legal_moves, squares, NSQ

EMPTY_TOK, MM, MK, OM, OK = 0, 1, 2, 3, 4
BOARD_VOCAB = 5
BOARD_TOKEN_NAMES = ("EMPTY", "MM", "MK", "OM", "OK")

SQUARE_TOKENS = tuple(f"<SQ_{s}>" for s in range(1, NSQ + 1))
PIECE_TOKENS = ("<PIECE_MM>", "<PIECE_MK>", "<PIECE_OM>", "<PIECE_OK>")
COLOUR_TOKENS = ("<WHITE>", "<BLACK>")
SPECIAL_TOKENS = SQUARE_TOKENS + PIECE_TOKENS + COLOUR_TOKENS

_PIECE_TOKEN_BY_ID = {MM: "<PIECE_MM>", MK: "<PIECE_MK>", OM: "<PIECE_OM>", OK: "<PIECE_OK>"}
_SQ_RE = re.compile(r"<SQ_(\d{1,2})>")


# ---------------------------------------------------------------------------
# board tokens
# ---------------------------------------------------------------------------

def encode_board(pos: Position) -> list[int]:
    """32 ints, square order, from the mover's point of view."""
    p = pos.pov()
    out = [EMPTY_TOK] * NSQ
    for s in squares(p.wm):
        out[s - 1] = MM
    for s in squares(p.wk):
        out[s - 1] = MK
    for s in squares(p.bm):
        out[s - 1] = OM
    for s in squares(p.bk):
        out[s - 1] = OK
    return out


def decode_board(ids: Iterable[int], turn: int = WHITE) -> Position:
    """Inverse of :func:`encode_board`.  ``turn`` is the real side to move;
    the POV rotation is undone when it is Black."""
    ids = list(ids)
    if len(ids) != NSQ:
        raise ValueError(f"expected {NSQ} ids, got {len(ids)}")
    wm = wk = bm = bk = 0
    for i, t in enumerate(ids):
        b = bit(i + 1)
        if t == MM: wm |= b
        elif t == MK: wk |= b
        elif t == OM: bm |= b
        elif t == OK: bk |= b
        elif t != EMPTY_TOK:
            raise ValueError(f"bad board token {t} at square {i + 1}")
    pov = Position(wm, wk, bm, bk, WHITE)
    if turn == WHITE:
        return pov
    # undo: rotate and swap colours
    return Position(_rot(pov.bm), _rot(pov.bk), _rot(pov.wm), _rot(pov.wk), BLACK)


def _rot(mask: int) -> int:
    out = 0
    for s in squares(mask):
        out |= bit(33 - s)
    return out


# ---------------------------------------------------------------------------
# POV square / move conversion
# ---------------------------------------------------------------------------

def pov_square(s: int, turn: int) -> int:
    return s if turn == WHITE else 33 - s


def pov_move(mv: Move, turn: int) -> Move:
    """Absolute move -> POV move (identity when White is to move)."""
    if turn == WHITE:
        return mv
    return Move(tuple(33 - s for s in mv.path), tuple(33 - s for s in mv.captured))


unpov_move = pov_move   # the map is an involution


# ---------------------------------------------------------------------------
# text tokens
# ---------------------------------------------------------------------------

def square_token(s: int) -> str:
    if not 1 <= s <= NSQ:
        raise ValueError(f"square out of range: {s}")
    return SQUARE_TOKENS[s - 1]


def piece_token(board_id: int) -> str:
    return _PIECE_TOKEN_BY_ID[board_id]


def move_to_text(mv: Move) -> str:
    """``<SQ_21><SQ_17>`` for a quiet move, ``<SQ_22>x<SQ_13>x<SQ_6>`` for a
    capture (full landing path, as :mod:`dq.rules` prints it)."""
    sep = "x" if mv.is_capture else ""
    return sep.join(square_token(s) for s in mv.path)


def text_to_move(text: str, legal: Iterable[Move]) -> Move:
    """Parse ``move_to_text`` output (or any string of ``<SQ_n>`` tokens with
    optional ``x``/``-`` separators) against ``legal``.  Raises ValueError
    if zero or several legal moves match."""
    sq = [int(m) for m in _SQ_RE.findall(text)]
    if len(sq) < 2:
        raise ValueError(f"bad move text {text!r}")
    want_capture = "x" in text
    legal = list(legal)
    cands = [mv for mv in legal if mv.is_capture == want_capture and list(mv.path) == sq]
    if len(cands) == 1:
        return cands[0]
    # tolerate the PDN short form start..end
    cands = [mv for mv in legal if mv.is_capture == want_capture and mv.start == sq[0] and mv.end == sq[-1]]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ValueError(f"illegal move text {text!r}")
    raise ValueError(f"ambiguous move text {text!r}")


def board_to_text(pos: Position) -> str:
    """Dictionary-form board for the *no-encoder* ablation (PLAN §5 Phase 2):
    POV-relative, one ``<PIECE_x><SQ_n>`` pair per piece, square order."""
    ids = encode_board(pos)
    parts = [f"{piece_token(t)}{square_token(i + 1)}" for i, t in enumerate(ids) if t != EMPTY_TOK]
    return " ".join(parts) if parts else "<EMPTY_BOARD>"


def legal_moves_pov(pos: Position) -> list[Move]:
    """Legal moves of ``pos`` expressed in POV squares."""
    return [pov_move(mv, pos.turn) for mv in legal_moves(pos)]


__all__ = [
    "EMPTY_TOK", "MM", "MK", "OM", "OK", "BOARD_VOCAB", "BOARD_TOKEN_NAMES",
    "SQUARE_TOKENS", "PIECE_TOKENS", "COLOUR_TOKENS", "SPECIAL_TOKENS",
    "encode_board", "decode_board", "pov_square", "pov_move", "unpov_move",
    "square_token", "piece_token", "move_to_text", "text_to_move", "board_to_text", "legal_moves_pov",
]
