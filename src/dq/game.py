"""Game loop with lidraughts-compatible draw adjudication for Brazilian.

Brazilian on lidraughts inherits the Russian draw rules (modules/draughts/
src/main/variant/{Brazilian,Russian}.scala).  Rule numbers are the FMJD /
Russian federation articles the source cites.  Counts below are **plies**
(the source stores 10/30/60/120 and its comments say 5/15/30/60 moves).

* 7.2.3  the same position with the same side to move occurs a third time.
* 7.2.4  three or more pieces, all kings, vs a lone king: 30 plies.
* 7.2.5  both sides have kings, more than 7 pieces: 30 consecutive king
         moves with no capture and no man move.
* 7.2.6  both sides have kings, 4-5 pieces: 60 plies; 6-7 pieces: 120
         plies; counter resets only on a capture or a promotion.
* 7.2.7  three pieces (>=1 king) vs a lone king that sits on the long
         diagonal (a1-h8) and is the only piece there: 30 plies, dropping
         to 10 once the lone king has made 10 king moves in the sequence.
* 7.2.8  one or two pieces (>=1 king) vs a lone king: 10 plies, never
         reset (except on the first promotion that creates the situation).

Which rule is active is re-evaluated after every move from the material on
the board, as lidraughts does; the counter carries over unless the move
resets it.  7.2.7's "10 king moves" is tracked per colour and reset with
the counter.  A hard cap (``max_plies``) ends runaway games as draws.

Result encoding (from White's point of view): ``1.0`` White wins, ``0.0``
Black wins, ``0.5`` draw.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from dq.rules import BLACK, WHITE, Move, Position, apply_move, bit, legal_moves, squares, zobrist

Policy = Callable[[Position, list[Move]], Move]

LONG_DIAGONAL = 0
for _s in (29, 25, 22, 18, 15, 11, 8, 4):   # a1 b2 c3 d4 e5 f6 g7 h8
    LONG_DIAGONAL |= bit(_s)


@dataclass
class DrawRule:
    limit: int                 # plies
    reset_on_man_move: bool    # 7.2.5
    reset_on_capture: bool
    reset_on_promotion: bool
    article: str


def _lone_king_rule(strong_pieces: int, strong_kings: int, weak_king_sq: int, pos: Position,
                    weak_king_moves: int) -> Optional[DrawRule]:
    if strong_pieces <= 2 and strong_kings >= 1:
        return DrawRule(10, False, False, False, "7.2.8")
    if strong_pieces == 3 and strong_kings >= 1 and (bit(weak_king_sq) & LONG_DIAGONAL) \
            and bin(pos.occupied & LONG_DIAGONAL).count("1") == 1:
        return DrawRule(10 if weak_king_moves >= 10 else 30, False, False, False, "7.2.7")
    if strong_pieces >= 3 and strong_kings == strong_pieces:
        return DrawRule(30, False, False, False, "7.2.4")
    return None


def active_draw_rule(pos: Position, king_moves: tuple[int, int]) -> Optional[DrawRule]:
    wp = bin(pos.wm | pos.wk).count("1"); wk = bin(pos.wk).count("1")
    bp = bin(pos.bm | pos.bk).count("1"); bk = bin(pos.bk).count("1")
    if wp == 1 and wk == 1:
        r = _lone_king_rule(bp, bk, squares(pos.wk)[0], pos, king_moves[WHITE])
        if r:
            return r
    if bp == 1 and bk == 1:
        r = _lone_king_rule(wp, wk, squares(pos.bk)[0], pos, king_moves[BLACK])
        if r:
            return r
    if wk >= 1 and bk >= 1:
        total = wp + bp
        if 6 <= total <= 7:
            return DrawRule(120, False, True, True, "7.2.6")
        if 4 <= total <= 5:
            return DrawRule(60, False, True, True, "7.2.6")
        return DrawRule(30, True, True, True, "7.2.5")
    return None


def draw_ply_limit(pos: Position) -> Optional[int]:
    """Ply limit of the draw rule that would be active at ``pos`` with fresh
    counters, or None if no counting rule applies.  Used to discount tablebase
    wins that cannot be converted before adjudication (optimistic about the
    counter's current value; pessimistic about resets by later captures)."""
    r = active_draw_rule(pos, (0, 0))
    return r.limit if r else None


@dataclass
class GameState:
    pos: Position
    history: list[tuple[Position, Move]] = field(default_factory=list)
    seen: Counter = field(default_factory=Counter)
    draw_counter: int = 0
    king_moves: list[int] = field(default_factory=lambda: [0, 0])
    rule: Optional[DrawRule] = None
    result: Optional[float] = None
    reason: str = ""

    def __post_init__(self) -> None:
        self.seen[zobrist(self.pos)] += 1
        self.rule = active_draw_rule(self.pos, (0, 0))

    @property
    def over(self) -> bool:
        return self.result is not None

    def legal(self) -> list[Move]:
        return legal_moves(self.pos)

    def push(self, mv: Move) -> None:
        if self.over:
            raise ValueError("game is over")
        pos = self.pos
        colour = pos.turn
        was_man = bool(pos.men(colour) & bit(mv.start))
        promoted = was_man and bool(bit(mv.end) & (0b1111 if colour == WHITE else 0b1111 << 28))
        new = apply_move(pos, mv)
        self.history.append((pos, mv))
        self.pos = new

        # --- draw counters (lidraughts order: evaluate rule on the new board) ---
        rule = active_draw_rule(new, tuple(self.king_moves))
        reset = False
        if rule is None:
            reset = True
        else:
            if mv.is_capture and rule.reset_on_capture:
                reset = True
            if promoted and rule.reset_on_promotion:
                reset = True
            if was_man and rule.reset_on_man_move:
                reset = True
            if rule.article == "7.2.8" and self.rule is not None and self.rule.article != "7.2.8":
                reset = True   # first entry into 7.2.8 starts its clock
            if rule.article == "7.2.8" and mv.is_capture:
                reset = True   # a capture that reduces into 7.2.8 restarts it
        if reset:
            self.draw_counter = 0
            self.king_moves = [0, 0]
        else:
            self.draw_counter += 1
            if not was_man:
                self.king_moves[colour] += 1
        self.rule = rule

        # --- termination ---
        h = zobrist(new)
        self.seen[h] += 1
        if not legal_moves(new):
            self.result = 1.0 if new.turn == BLACK else 0.0
            self.reason = "no moves"
        elif self.seen[h] >= 3:
            self.result, self.reason = 0.5, "7.2.3 threefold"
        elif rule is not None and self.draw_counter >= rule.limit:
            self.result, self.reason = 0.5, rule.article


def play_game(white: Policy, black: Policy, start: Position | None = None, max_plies: int = 400) -> GameState:
    g = GameState(start or Position.start())
    while not g.over:
        if len(g.history) >= max_plies:
            g.result, g.reason = 0.5, "max plies"
            break
        moves = g.legal()
        mv = (white if g.pos.turn == WHITE else black)(g.pos, moves)
        g.push(mv)
    return g


def random_policy(rng) -> Policy:
    return lambda pos, moves: rng.choice(moves)
