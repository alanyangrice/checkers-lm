"""Thin wrapper around pydraughts' Brazilian rules, exposing the same
``Position`` / ``Move`` types as :mod:`dq.rules`.

This is the *reference oracle*: it is slow (object-based, pickled copies)
and is used only to cross-check the native generator in tests and in
``scripts/crosscheck.py``.  Production code should use :mod:`dq.rules`.

pydraughts' ``Game`` layer uses the lidraughts numbering directly, so no
square translation is needed.  (Its higher-level ``Board`` class rotates
Brazilian boards to white-at-bottom numbering; we avoid that layer.)
"""
from __future__ import annotations

from draughts.core.game import Game as _Game

from dq.rules import BLACK, WHITE, Move, Position

_PD_WHITE, _PD_BLACK = 2, 1


class RefBoard:
    """A pydraughts-backed Brazilian game."""

    def __init__(self, start: Position | str | None = None) -> None:
        if start is None:
            start = Position.start()
        fen = start.fen() if isinstance(start, Position) else Position.from_fen(start).fen()
        self._game = _Game("brazilian", fen)

    # -- state ----------------------------------------------------------------
    def position(self) -> Position:
        return Position.from_fen(self._game.get_li_fen())

    @property
    def turn(self) -> int:
        return WHITE if self._game.whose_turn() == _PD_WHITE else BLACK

    def winner(self) -> int | str | None:
        """WHITE / BLACK / ``"draw"`` / None (game continues)."""
        w = self._game.get_winner()
        if w is None:
            return None
        if w == 0:
            return "draw"
        return WHITE if w == _PD_WHITE else BLACK

    # -- moves ----------------------------------------------------------------
    def legal_moves(self) -> list[Move]:
        moves, captures = self._game.legal_moves()
        out: list[Move] = []
        for bm, caps in zip(moves, captures):
            path = (bm[0][0],) + tuple(step[1] for step in bm)
            captured = tuple(c for c in caps if c is not None)
            out.append(Move(path, captured))
        return out

    def push(self, mv: Move) -> None:
        steps = [[a, b] for a, b in zip(mv.path, mv.path[1:])]
        self._game.push(steps)

    def pop(self) -> None:
        self._game.pop()


def ref_legal_moves(pos: Position) -> list[Move]:
    return RefBoard(pos).legal_moves()


def ref_apply(pos: Position, mv: Move) -> Position:
    b = RefBoard(pos)
    b.push(mv)
    return b.position()
