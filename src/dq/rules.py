"""Brazilian draughts rules: a small, dependency-free move generator.

Conventions
-----------
* Squares are numbered 1..32 in the lidraughts / pydraughts-internal scheme:
  square 1 is b8 (top-left dark square seen from White), 4 is h8, 5 is a7,
  ..., 29 is a1, 32 is g1.  Black men start on 1-12, White men on 21-32.
  White moves first and moves "up" the board, i.e. toward square 1.
* Colours are ``WHITE = 0`` and ``BLACK = 1``.
* A ``Position`` is immutable.  Boards are four 32-bit masks (bit ``s-1``
  is square ``s``) plus the side to move.
* A ``Move`` is the full landing path of the moving piece plus the squares
  of the captured pieces in capture order.  With flying kings the same set
  of captured pieces can be taken via different landing squares; those are
  distinct moves.

Rules implemented (Brazilian = international rules on 8x8):
* men move one step diagonally forward, capture one step diagonally in all
  four directions by jumping to the empty square directly behind;
* kings slide any distance and, when capturing, may land on any empty
  square beyond the captured piece before continuing;
* captures are mandatory and the sequence(s) capturing the most pieces
  must be chosen (kings and men count equally);
* a piece may not be jumped twice; captured pieces stay on the board
  (as obstacles) until the sequence is complete;
* a man that passes through the back rank mid-capture does not promote and
  keeps capturing as a man; it promotes only if the sequence ends there;
* the side to move with no legal move loses.

Draw adjudication (repetition, king-only move counts) is deliberately *not*
part of this module; it belongs with the self-play / game layer.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

WHITE = 0
BLACK = 1
COLOR_NAMES = ("white", "black")

NSQ = 32
FILES = "abcdefgh"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def square_to_rf(s: int) -> tuple[int, int]:
    """Square 1..32 -> (rank index 0..7 with 0 = rank 1, file index 0..7)."""
    i = s - 1
    row = i // 4            # 0 = top row (rank 8)
    r = 7 - row
    f = 2 * (i % 4) + (1 if row % 2 == 0 else 0)
    return r, f


def rf_to_square(r: int, f: int) -> int:
    """(rank index, file index) -> square 1..32.  Raises for light squares."""
    if not (0 <= r < 8 and 0 <= f < 8) or (r + f) % 2:
        raise ValueError(f"not a playable square: rank {r} file {f}")
    row = 7 - r
    return row * 4 + f // 2 + 1


def square_to_alg(s: int) -> str:
    r, f = square_to_rf(s)
    return f"{FILES[f]}{r + 1}"


def alg_to_square(a: str) -> int:
    a = a.strip().lower()
    if len(a) != 2 or a[0] not in FILES or not a[1].isdigit():
        raise ValueError(f"bad algebraic square {a!r}")
    return rf_to_square(int(a[1]) - 1, FILES.index(a[0]))


# Direction d: (d_rank, d_file).  WHITE moves toward higher rank index.
DIRS = ((+1, -1), (+1, +1), (-1, -1), (-1, +1))
FORWARD_DIRS = ((0, 1), (2, 3))   # indexed by colour


def _build_next() -> tuple[tuple[int, ...], ...]:
    table = []
    for s in range(1, NSQ + 1):
        r, f = square_to_rf(s)
        row = []
        for dr, df in DIRS:
            rr, ff = r + dr, f + df
            row.append(rf_to_square(rr, ff) - 1 if 0 <= rr < 8 and 0 <= ff < 8 else -1)
        table.append(tuple(row))
    return tuple(table)


NEXT = _build_next()          # NEXT[i][d] -> square index 0..31 or -1
BACK_RANK = (0b1111, 0b1111 << 28)   # promotion squares for WHITE (1-4) / BLACK (29-32)
ALL = (1 << NSQ) - 1


def bit(s: int) -> int:
    return 1 << (s - 1)


def squares(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length())
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

# piece codes
EMPTY, WM, WK, BM, BK = 0, 1, 2, 3, 4
PIECE_CHARS = {WM: "w", WK: "W", BM: "b", BK: "B", EMPTY: "."}


@dataclass(frozen=True, slots=True)
class Position:
    wm: int = 0
    wk: int = 0
    bm: int = 0
    bk: int = 0
    turn: int = WHITE

    # -- construction -----------------------------------------------------
    @staticmethod
    def start() -> "Position":
        return Position(wm=0b1111_1111_1111 << 20, bm=0b1111_1111_1111, turn=WHITE)

    @staticmethod
    def from_fen(fen: str) -> "Position":
        """Parse ``W:W21,22,K23:B1,2`` (lidraughts style, numeric or algebraic
        squares, ranges like ``21-32`` allowed)."""
        parts = fen.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"bad FEN {fen!r}")
        turn = {"W": WHITE, "B": BLACK}.get(parts[0].upper())
        if turn is None:
            raise ValueError(f"bad side to move in {fen!r}")
        masks = {WM: 0, WK: 0, BM: 0, BK: 0}
        for part in parts[1:]:
            if not part or part[0].upper() not in "WB":
                raise ValueError(f"bad side block {part!r} in {fen!r}")
            colour = WHITE if part[0].upper() == "W" else BLACK
            for item in filter(None, part[1:].split(",")):
                king = item[0].upper() == "K"
                if king:
                    item = item[1:]
                if "-" in item:
                    a, b = item.split("-")
                    rng = range(_parse_square(a), _parse_square(b) + 1)
                else:
                    rng = (_parse_square(item),)
                code = (WK if king else WM) if colour == WHITE else (BK if king else BM)
                for s in rng:
                    masks[code] |= bit(s)
        pos = Position(masks[WM], masks[WK], masks[BM], masks[BK], turn)
        pos.validate()
        return pos

    def validate(self) -> None:
        ms = (self.wm, self.wk, self.bm, self.bk)
        if (self.wm | self.wk | self.bm | self.bk) & ~ALL:
            raise ValueError("piece off board")
        if sum(ms) != (self.wm | self.wk | self.bm | self.bk):
            raise ValueError("two pieces on one square")
        if self.wm & BACK_RANK[WHITE] or self.bm & BACK_RANK[BLACK]:
            raise ValueError("man on its promotion rank")

    # -- queries ------------------------------------------------------------
    def piece_at(self, s: int) -> int:
        b = bit(s)
        if self.wm & b:
            return WM
        if self.wk & b:
            return WK
        if self.bm & b:
            return BM
        if self.bk & b:
            return BK
        return EMPTY

    def men(self, colour: int) -> int:
        return self.wm if colour == WHITE else self.bm

    def kings(self, colour: int) -> int:
        return self.wk if colour == WHITE else self.bk

    def pieces(self, colour: int) -> int:
        return self.men(colour) | self.kings(colour)

    @property
    def occupied(self) -> int:
        return self.wm | self.wk | self.bm | self.bk

    @property
    def opponent(self) -> int:
        return self.turn ^ 1

    def fen(self) -> str:
        def block(men: int, kings: int) -> str:
            items = [(s, "") for s in squares(men)] + [(s, "K") for s in squares(kings)]
            return ",".join(f"{k}{s}" for s, k in sorted(items))
        return f"{'WB'[self.turn]}:W{block(self.wm, self.wk)}:B{block(self.bm, self.bk)}"

    def ascii(self) -> str:
        rows = []
        for r in range(7, -1, -1):
            cells = []
            for f in range(8):
                cells.append(PIECE_CHARS[self.piece_at(rf_to_square(r, f))] if (r + f) % 2 == 0 else " ")
            rows.append(f"{r + 1} " + " ".join(cells))
        rows.append("  " + " ".join(FILES))
        return "\n".join(rows)

    def pov(self) -> "Position":
        """Rotate 180 degrees and swap colours so the side to move is White
        moving toward square 1.  Identity when White is to move."""
        if self.turn == WHITE:
            return self
        return Position(_rot(self.bm), _rot(self.bk), _rot(self.wm), _rot(self.wk), WHITE)

    def __str__(self) -> str:
        return self.fen()


def _parse_square(tok: str) -> int:
    tok = tok.strip()
    s = int(tok) if tok.isdigit() else alg_to_square(tok)
    if not 1 <= s <= NSQ:
        raise ValueError(f"square out of range: {tok}")
    return s


def _rot(mask: int) -> int:
    """180-degree rotation: square s -> 33 - s."""
    out = 0
    for s in squares(mask):
        out |= bit(33 - s)
    return out


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Move:
    path: tuple[int, ...]            # squares visited, starting with the origin
    captured: tuple[int, ...] = ()   # captured squares in capture order

    @property
    def start(self) -> int:
        return self.path[0]

    @property
    def end(self) -> int:
        return self.path[-1]

    @property
    def is_capture(self) -> bool:
        return bool(self.captured)

    def __str__(self) -> str:
        sep = "x" if self.captured else "-"
        return sep.join(str(s) for s in self.path)

    def alg(self) -> str:
        sep = "x" if self.captured else "-"
        return sep.join(square_to_alg(s) for s in self.path)

    def __lt__(self, other: "Move") -> bool:
        return (self.path, self.captured) < (other.path, other.captured)


def _sequences(pos: Position, s: int, king: bool) -> list[Move]:
    """All *complete* capture sequences for the piece on square ``s``."""
    opp = pos.pieces(pos.opponent)
    occ = pos.occupied & ~bit(s)      # the mover is lifted off its origin
    out: list[Move] = []

    def man(i: int, path: list[int], cap: int, caps: list[int]) -> None:
        found = False
        for d in range(4):
            j = NEXT[i][d]
            if j < 0 or not (opp & (1 << j)) or cap & (1 << j):
                continue
            k = NEXT[j][d]
            if k < 0 or occ & (1 << k):
                continue
            found = True
            path.append(k + 1); caps.append(j + 1)
            man(k, path, cap | (1 << j), caps)
            path.pop(); caps.pop()
        if not found and caps:
            out.append(Move(tuple(path), tuple(caps)))

    def kng(i: int, path: list[int], cap: int, caps: list[int]) -> None:
        found = False
        for d in range(4):
            j = NEXT[i][d]
            while j >= 0 and not occ & (1 << j):
                j = NEXT[j][d]
            if j < 0 or not (opp & (1 << j)) or cap & (1 << j):
                continue
            k = NEXT[j][d]
            while k >= 0 and not occ & (1 << k):
                found = True
                path.append(k + 1); caps.append(j + 1)
                kng(k, path, cap | (1 << j), caps)
                path.pop(); caps.pop()
                k = NEXT[k][d]
        if not found and caps:
            out.append(Move(tuple(path), tuple(caps)))

    (kng if king else man)(s - 1, [s], 0, [])
    return out


def capture_sequences(pos: Position) -> list[Move]:
    """Every complete capture sequence for the side to move, regardless of
    length (i.e. before the max-capture rule is applied)."""
    out: list[Move] = []
    for s in squares(pos.men(pos.turn)):
        out.extend(_sequences(pos, s, king=False))
    for s in squares(pos.kings(pos.turn)):
        out.extend(_sequences(pos, s, king=True))
    return out


def max_capture_filter(moves: Iterable[Move]) -> list[Move]:
    moves = list(moves)
    if not moves:
        return []
    m = max(len(mv.captured) for mv in moves)
    return [mv for mv in moves if len(mv.captured) == m]


def quiet_moves(pos: Position) -> list[Move]:
    occ = pos.occupied
    out: list[Move] = []
    for s in squares(pos.men(pos.turn)):
        for d in FORWARD_DIRS[pos.turn]:
            j = NEXT[s - 1][d]
            if j >= 0 and not occ & (1 << j):
                out.append(Move((s, j + 1)))
    for s in squares(pos.kings(pos.turn)):
        for d in range(4):
            j = NEXT[s - 1][d]
            while j >= 0 and not occ & (1 << j):
                out.append(Move((s, j + 1)))
                j = NEXT[j][d]
    return out


def legal_moves(pos: Position) -> list[Move]:
    caps = capture_sequences(pos)
    if caps:
        return max_capture_filter(caps)
    return quiet_moves(pos)


def apply_move(pos: Position, mv: Move) -> Position:
    """Apply ``mv`` (assumed legal) and return the new position."""
    colour = pos.turn
    men, kings = pos.men(colour), pos.kings(colour)
    omen, okings = pos.men(colour ^ 1), pos.kings(colour ^ 1)
    sb, eb = bit(mv.start), bit(mv.end)
    if men & sb:
        men &= ~sb
        if eb & BACK_RANK[colour]:
            kings |= eb
        else:
            men |= eb
    elif kings & sb:
        kings = (kings & ~sb) | eb
    else:
        raise ValueError(f"no {COLOR_NAMES[colour]} piece on {mv.start}")
    for c in mv.captured:
        cb = bit(c)
        omen &= ~cb
        okings &= ~cb
    if colour == WHITE:
        return Position(men, kings, omen, okings, BLACK)
    return Position(omen, okings, men, kings, WHITE)


def is_terminal(pos: Position) -> bool:
    return not legal_moves(pos)


def winner(pos: Position) -> int | None:
    """Colour that has won by the no-moves rule, or None if play continues."""
    return pos.opponent if is_terminal(pos) else None


def parse_move(text: str, legal: Iterable[Move]) -> Move:
    """Resolve a move string against a list of legal moves.

    Accepts the full landing path (``22x13x6``), the PDN short form
    (``22x6`` or ``21-17``), algebraic squares (``d6xb4xd2``), and a path with
    only the direction-change squares as PDN allows.  Raises ``ValueError``
    if the string matches zero or more than one legal move.
    """
    legal = list(legal)
    t = text.strip().lower().replace(":", "x")
    sep = "x" if "x" in t else "-"
    toks = [tok for tok in t.split(sep) if tok]
    if len(toks) < 2:
        raise ValueError(f"bad move string {text!r}")
    sq = [_parse_square(tok) for tok in toks]
    want_capture = sep == "x"
    cands = [mv for mv in legal if mv.is_capture == want_capture and mv.start == sq[0] and mv.end == sq[-1]
             and _is_subsequence(sq, mv.path)]
    if len(cands) == 1:
        return cands[0]
    exact = [mv for mv in cands if list(mv.path) == sq]
    if len(exact) == 1:
        return exact[0]
    if not cands:
        raise ValueError(f"illegal move {text!r}")
    raise ValueError(f"ambiguous move {text!r}: " + ", ".join(map(str, cands)))


def _is_subsequence(sub: list[int], full: tuple[int, ...]) -> bool:
    it = iter(full)
    return all(any(x == y for y in it) for x in sub)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

_rng = random.Random(0x5EED_D2A0)
ZOBRIST = [[_rng.getrandbits(64) for _ in range(NSQ)] for _ in range(5)]
ZOBRIST_BLACK = _rng.getrandbits(64)


def zobrist(pos: Position) -> int:
    h = ZOBRIST_BLACK if pos.turn == BLACK else 0
    for code, mask in ((WM, pos.wm), (WK, pos.wk), (BM, pos.bm), (BK, pos.bk)):
        row = ZOBRIST[code]
        for s in squares(mask):
            h ^= row[s - 1]
    return h


# ---------------------------------------------------------------------------
# Random positions (for tests and curriculum sampling)
# ---------------------------------------------------------------------------

def random_position(rng: random.Random, n_white: int, n_black: int, king_prob: float = 0.25,
                    turn: int | None = None) -> Position:
    """Uniform-ish random legal placement.  Men are never placed on their own
    promotion rank."""
    sq = list(range(1, NSQ + 1))
    rng.shuffle(sq)
    masks = {WM: 0, WK: 0, BM: 0, BK: 0}
    idx = 0
    for colour, n in ((WHITE, n_white), (BLACK, n_black)):
        placed = 0
        while placed < n:
            s = sq[idx]; idx += 1
            king = rng.random() < king_prob
            if not king and bit(s) & BACK_RANK[colour]:
                king = True
            code = (WK if king else WM) if colour == WHITE else (BK if king else BM)
            masks[code] |= bit(s)
            placed += 1
    t = rng.randrange(2) if turn is None else turn
    return Position(masks[WM], masks[WK], masks[BM], masks[BK], t)


__all__ = [
    "WHITE", "BLACK", "WM", "WK", "BM", "BK", "EMPTY", "NSQ", "NEXT", "DIRS", "BACK_RANK",
    "Position", "Move", "legal_moves", "capture_sequences", "max_capture_filter", "quiet_moves",
    "apply_move", "is_terminal", "winner", "parse_move", "zobrist", "random_position",
    "square_to_alg", "alg_to_square", "square_to_rf", "rf_to_square", "squares", "bit",
]
