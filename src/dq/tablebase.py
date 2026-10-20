"""Retrograde endgame tablebases for Brazilian draughts (WDL + distance).

Every stored position is *POV-normalised*: White to move (``Position.pov()``).
A real position with Black to move is looked up through its rotation.

Positions are grouped into **units** keyed by ``(total_pieces, total_men)``.
A move never increases total pieces, and a promotion decreases total men,
so a unit's children lie in the same unit (quiet moves, incl. mirror
classes) or in a unit that sorts earlier under ``(total, men)``.  Units
are solved in that order; within a unit the standard retrograde algorithm
runs over a CSR parent graph with depth buckets, giving exact
distance-to-win / distance-to-loss in plies (ignoring draw-by-rule).

Values: ``WIN = 1``, ``LOSS = -1``, ``DRAW = 0`` for the side to move;
``dtm`` is the number of plies to the terminal position under optimal play
(``0`` for a position with no legal move, which is a loss).

This pure-Python/numpy version is for validation and small tables (<= 4
pieces in minutes).  Larger tables need the inner loops compiled; the
on-disk format (``keys``, ``wdl``, ``dtm`` per unit) is kept simple so a
faster builder can write the same files.
"""
from __future__ import annotations

import itertools
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dq.rules import BACK_RANK, BLACK, NSQ, WHITE, Position, apply_move, bit, legal_moves, squares

WIN, DRAW, LOSS = 1, 0, -1
UNKNOWN = 2  # internal

_WM_SQ = [s for s in range(1, NSQ + 1) if not bit(s) & BACK_RANK[WHITE]]   # 5..32
_BM_SQ = [s for s in range(1, NSQ + 1) if not bit(s) & BACK_RANK[BLACK]]   # 1..28
_ALL_SQ = list(range(1, NSQ + 1))


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

def pack(pos: Position) -> int:
    """Pack a POV (White-to-move) position: 32-bit occupancy, then 2 bits per
    piece in ascending-square order (0 wm, 1 wk, 2 bm, 3 bk).  <= 7 pieces
    fits in 46 bits."""
    assert pos.turn == WHITE
    occ = pos.occupied
    key = occ
    shift = 32
    for s in squares(occ):
        b = bit(s)
        t = 0 if pos.wm & b else 1 if pos.wk & b else 2 if pos.bm & b else 3
        key |= t << shift
        shift += 2
    return key


def unpack(key: int) -> Position:
    occ = key & 0xFFFFFFFF
    wm = wk = bm = bk = 0
    shift = 32
    for s in squares(occ):
        t = (key >> shift) & 3
        shift += 2
        b = bit(s)
        if t == 0: wm |= b
        elif t == 1: wk |= b
        elif t == 2: bm |= b
        else: bk |= b
    return Position(wm, wk, bm, bk, WHITE)


def unit_of(pos: Position) -> tuple[int, int]:
    p = pos.pov()
    men = bin(p.wm | p.bm).count("1")
    total = men + bin(p.wk | p.bk).count("1")
    return total, men


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

def _placements(wm: int, wk: int, bm: int, bk: int):
    """Yield packed keys of every POV position with these piece counts."""
    for wms in itertools.combinations(_WM_SQ, wm):
        used = 0
        for s in wms: used |= bit(s)
        rest_k = [s for s in _ALL_SQ if not used & bit(s)]
        for wks in itertools.combinations(rest_k, wk):
            used2 = used
            for s in wks: used2 |= bit(s)
            rest_bm = [s for s in _BM_SQ if not used2 & bit(s)]
            for bms in itertools.combinations(rest_bm, bm):
                used3 = used2
                for s in bms: used3 |= bit(s)
                rest_bk = [s for s in _ALL_SQ if not used3 & bit(s)]
                for bks in itertools.combinations(rest_bk, bk):
                    m_wm = m_wk = m_bm = m_bk = 0
                    for s in wms: m_wm |= bit(s)
                    for s in wks: m_wk |= bit(s)
                    for s in bms: m_bm |= bit(s)
                    for s in bks: m_bk |= bit(s)
                    yield pack(Position(m_wm, m_wk, m_bm, m_bk, WHITE))


def enumerate_unit(total: int, men: int) -> np.ndarray:
    """Sorted int64 keys of all POV positions in unit (total, men), with both
    sides having at least one piece."""
    keys = array("q")
    kings = total - men
    for wm in range(men + 1):
        bm = men - wm
        for wk in range(kings + 1):
            bk = kings - wk
            if wm + wk == 0 or bm + bk == 0:
                continue
            keys.extend(_placements(wm, wk, bm, bk))
    arr = np.frombuffer(keys, dtype=np.int64).copy() if len(keys) else np.zeros(0, np.int64)
    arr.sort()
    return arr


def units_up_to(max_pieces: int) -> list[tuple[int, int]]:
    out = []
    for total in range(2, max_pieces + 1):
        for men in range(0, total + 1):
            out.append((total, men))
    return out


# ---------------------------------------------------------------------------
# tablebase
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    total: int
    men: int
    keys: np.ndarray                      # sorted int64
    wdl: np.ndarray                       # int8, WIN/DRAW/LOSS
    dtm: np.ndarray                       # int16, plies; 0 for terminal loss; -1 for draw

    def index(self, key: int) -> int:
        i = int(np.searchsorted(self.keys, key))
        if i >= len(self.keys) or self.keys[i] != key:
            raise KeyError(key)
        return i


@dataclass
class Tablebase:
    units: dict[tuple[int, int], Unit] = field(default_factory=dict)

    # -- lookup ------------------------------------------------------------
    def probe(self, pos: Position) -> tuple[int, int] | None:
        """(wdl, dtm) for the side to move, or None if not covered."""
        u = self.units.get(unit_of(pos))
        if u is None:
            return None
        p = pos.pov()
        try:
            i = u.index(pack(p))
        except KeyError:
            return None
        return int(u.wdl[i]), int(u.dtm[i])

    def covers(self, pos: Position) -> bool:
        return unit_of(pos) in self.units

    # -- build -------------------------------------------------------------
    def build(self, max_pieces: int, verbose: bool = False) -> None:
        for total, men in units_up_to(max_pieces):
            if (total, men) in self.units:
                continue
            self.units[(total, men)] = self._solve_unit(total, men, verbose)

    def _solve_unit(self, total: int, men: int, verbose: bool) -> Unit:
        keys = enumerate_unit(total, men)
        n = len(keys)
        if verbose:
            print(f"unit ({total},{men}): {n} positions", flush=True)
        wdl = np.full(n, UNKNOWN, np.int8)
        dtm = np.full(n, -1, np.int16)

        # per-position bookkeeping for the retrograde pass
        unresolved = np.zeros(n, np.int32)        # same-unit children not yet resolved
        max_child_dtm = np.full(n, -1, np.int32)  # max dtm over children known to be WIN
        # parents CSR (child -> parents) built from forward edges
        fwd_src = array("i"); fwd_dst = array("i")
        # scheduled candidates: depth -> list of (index, value)
        buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)

        key_list = keys.tolist()
        for i, key in enumerate(key_list):
            pos = unpack(key)
            moves = legal_moves(pos)
            if not moves:
                buckets[0].append((i, LOSS))
                continue
            n_same = 0
            static_min_loss = None     # min dtm among solved-unit children that are LOSS
            static_max_win = -1        # max dtm among solved-unit children that are WIN
            static_any_draw = False
            for mv in moves:
                child = apply_move(pos, mv).pov()
                if child.pieces(WHITE) == 0:
                    # opponent's last piece captured: child mover has no move
                    static_min_loss = 0
                    continue
                cu = unit_of(child)
                ck = pack(child)
                if cu == (total, men):
                    j = int(np.searchsorted(keys, ck))
                    assert keys[j] == ck, (pos.fen(), str(mv), child.fen())
                    fwd_src.append(i); fwd_dst.append(j)
                    n_same += 1
                else:
                    u = self.units[cu]
                    j = u.index(ck)
                    v, d = int(u.wdl[j]), int(u.dtm[j])
                    if v == LOSS:
                        static_min_loss = d if static_min_loss is None else min(static_min_loss, d)
                    elif v == WIN:
                        static_max_win = max(static_max_win, d)
                    else:
                        static_any_draw = True
            if static_min_loss is not None:
                buckets[static_min_loss + 1].append((i, WIN))
            unresolved[i] = n_same
            max_child_dtm[i] = static_max_win
            if static_any_draw or static_min_loss is not None:
                unresolved[i] = -1   # can never be a loss (has a drawing or losing child)
            elif n_same == 0:
                # all children static wins -> loss
                buckets[static_max_win + 1].append((i, LOSS))

        # build parents CSR
        src = np.frombuffer(fwd_src, dtype=np.int32) if len(fwd_src) else np.zeros(0, np.int32)
        dst = np.frombuffer(fwd_dst, dtype=np.int32) if len(fwd_dst) else np.zeros(0, np.int32)
        order = np.argsort(dst, kind="stable")
        par_idx = src[order]
        counts = np.bincount(dst, minlength=n) if n else np.zeros(0, np.int64)
        par_ptr = np.zeros(n + 1, np.int64)
        np.cumsum(counts, out=par_ptr[1:])

        depth = 0
        resolved = 0
        while buckets:
            if depth not in buckets:
                if depth > max(buckets):
                    break
                depth += 1
                continue
            items = buckets.pop(depth)
            for i, v in items:
                if wdl[i] != UNKNOWN:
                    continue           # already resolved at a smaller depth
                wdl[i] = v
                dtm[i] = depth
                resolved += 1
                for p in par_idx[par_ptr[i]:par_ptr[i + 1]]:
                    if wdl[p] != UNKNOWN:
                        continue
                    if v == LOSS:
                        buckets[depth + 1].append((int(p), WIN))
                    else:  # child is a WIN for its mover -> bad for parent
                        if unresolved[p] < 0:
                            continue   # parent has a drawing child; can't become loss
                        unresolved[p] -= 1
                        if depth > max_child_dtm[p]:
                            max_child_dtm[p] = depth
                        if unresolved[p] == 0:
                            buckets[int(max_child_dtm[p]) + 1].append((int(p), LOSS))
            depth += 1

        wdl[wdl == UNKNOWN] = DRAW
        if verbose:
            print(f"  win={int((wdl == WIN).sum())} loss={int((wdl == LOSS).sum())} "
                  f"draw={int((wdl == DRAW).sum())} max_dtm={int(dtm.max()) if n else 0}", flush=True)
        return Unit(total, men, keys, wdl, dtm)

    # -- io ----------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        for (total, men), u in self.units.items():
            np.savez_compressed(d / f"tb_{total}_{men}.npz", keys=u.keys, wdl=u.wdl, dtm=u.dtm)

    @classmethod
    def load(cls, directory: str | Path) -> "Tablebase":
        tb = cls()
        for f in sorted(Path(directory).glob("tb_*_*.npz")):
            _, total, men = f.stem.split("_")
            z = np.load(f)
            tb.units[(int(total), int(men))] = Unit(int(total), int(men), z["keys"], z["wdl"], z["dtm"])
        return tb


# ---------------------------------------------------------------------------
# brute-force reference (for tests)
# ---------------------------------------------------------------------------

def minimax(pos: Position, max_depth: int, _memo: dict | None = None) -> tuple[int, int] | None:
    """Depth-limited negamax returning (wdl, dtm) when the outcome is forced
    within ``max_depth`` plies, else None (unknown).  Exponential; tests only."""
    if _memo is None:
        _memo = {}
    key = (pos.pov().fen(), max_depth)
    if key in _memo:
        return _memo[key]
    moves = legal_moves(pos)
    if not moves:
        res = (LOSS, 0)
        _memo[key] = res
        return res
    if max_depth == 0:
        _memo[key] = None
        return None
    best_win = None
    all_known_loss = True
    worst_loss = -1
    for mv in moves:
        r = minimax(apply_move(pos, mv), max_depth - 1, _memo)
        if r is None:
            all_known_loss = False
            continue
        v, d = r
        if v == LOSS:
            if best_win is None or d + 1 < best_win:
                best_win = d + 1
        else:
            all_known_loss = False if v == DRAW else all_known_loss
            if v == WIN:
                worst_loss = max(worst_loss, d + 1)
    if best_win is not None:
        res = (WIN, best_win)
    elif all_known_loss:
        res = (LOSS, worst_loss)
    else:
        res = None
    _memo[key] = res
    return res
