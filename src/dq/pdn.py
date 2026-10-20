"""Minimal PDN reader for lidraughts Brazilian exports, and a replayer that
checks every move against :mod:`dq.rules`.

lidraughts exports Brazilian games with algebraic moves (``c3-d4``,
``h4xf2``) and a ``[FEN]`` header in algebraic squares.  Capture moves are
written start x end only (PDN short form); :func:`dq.rules.parse_move`
resolves them against the legal-move list.  Results are ``2-0`` / ``0-2`` /
``1-1`` (draughts scoring), with ``*`` for unfinished games.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from dq.rules import BLACK, WHITE, Move, Position, apply_move, legal_moves, parse_move

_HDR = re.compile(r'^\[(\w+)\s+"([^"]*)"\]\s*$', re.M)
_MOVENUM = re.compile(r"\d+\.(\.\.)?")
_COMMENT = re.compile(r"\{[^}]*\}")
_RESULT = {"2-0": WHITE, "0-2": BLACK, "1-1": "draw", "*": None, "1-0": WHITE, "0-1": BLACK, "1/2-1/2": "draw"}


@dataclass
class Game:
    headers: dict[str, str]
    moves: list[str]                 # move strings as written
    result: int | str | None         # WHITE / BLACK / "draw" / None

    @property
    def site(self) -> str:
        return self.headers.get("Site", "")


def split_games(text: str) -> list[str]:
    """Split a multi-game PDN file into per-game chunks."""
    chunks, cur = [], []
    for line in text.splitlines():
        if line.startswith("[Event ") and cur and any(not l.startswith("[") and l.strip() for l in cur):
            chunks.append("\n".join(cur)); cur = []
        cur.append(line)
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if c.strip()]


def parse_game(chunk: str) -> Game:
    headers = dict(_HDR.findall(chunk))
    body = "\n".join(l for l in chunk.splitlines() if not l.startswith("["))
    body = _COMMENT.sub(" ", body)
    body = _MOVENUM.sub(" ", body)
    toks = body.split()
    result = None
    if toks and toks[-1] in _RESULT:
        result = _RESULT[toks.pop()]
    return Game(headers, toks, result)


def read_pdn(text: str) -> list[Game]:
    return [parse_game(c) for c in split_games(text)]


@dataclass
class Replay:
    ok: bool
    plies: int
    final: Position
    error: str | None = None
    positions: list[Position] = field(default_factory=list)
    played: list[Move] = field(default_factory=list)


def _candidates(tok: str, legal: list[Move]) -> list[Move]:
    """All legal moves matching a PDN move string (start/end + any listed
    intermediate squares).  lidraughts' short form ``b8xg3`` can match several
    capture paths, sometimes taking different pieces; the caller
    disambiguates by continuation."""
    try:
        return [parse_move(tok, legal)]
    except ValueError as e:
        if not str(e).startswith("ambiguous"):
            return []
    t = tok.strip().lower().replace(":", "x")
    sep = "x" if "x" in t else "-"
    sq = [_sq(x) for x in t.split(sep) if x]
    return [mv for mv in legal if mv.is_capture == (sep == "x") and mv.start == sq[0] and mv.end == sq[-1]]


def _sq(tok: str) -> int:
    from dq.rules import _parse_square
    return _parse_square(tok)


def replay(game: Game, keep: bool = False) -> Replay:
    """Play every move of ``game`` through the native rules.  ``ok`` is False
    on the first illegal move, with ``error`` describing it.  An ambiguous
    short-form capture is resolved by whichever reading lets the rest of the
    game replay legally (first such reading if several do)."""
    fen = game.headers.get("FEN")
    start = Position.from_fen(fen) if fen else Position.start()
    moves = game.moves

    def go(pos: Position, i: int, played: list[Move], positions: list[Position]) -> Replay:
        while i < len(moves):
            legal = legal_moves(pos)
            cands = _candidates(moves[i], legal)
            if not cands:
                return Replay(False, i, pos, f"ply {i} {moves[i]!r}: illegal", positions, played)
            if len(cands) == 1:
                pos = apply_move(pos, cands[0])
                played.append(cands[0])
                if keep:
                    positions.append(pos)
                i += 1
                continue
            # ambiguous: branch
            last_fail = None
            for mv in cands:
                r = go(apply_move(pos, mv), i + 1, played + [mv],
                       positions + ([apply_move(pos, mv)] if keep else []))
                if r.ok:
                    return r
                last_fail = r
            return last_fail  # type: ignore[return-value]
        return Replay(True, len(moves), pos, None, positions, played)

    return go(start, 0, [], [start] if keep else [])


def consistent_result(game: Game, rep: Replay) -> bool | None:
    """True/False if the recorded result can be checked against the final
    position, None if it cannot (resignation, time, draw agreement)."""
    if not rep.ok:
        return False
    term = game.headers.get("Termination", "")
    if game.result in (WHITE, BLACK) and term == "Normal":
        # a decisive 'Normal' termination is either no-moves or resignation;
        # if the loser has no legal moves the result must match
        if not legal_moves(rep.final):
            return rep.final.opponent == game.result
    return None
