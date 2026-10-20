"""Targeted cross-check of dq.rules against pydraughts on the shapes where
move generators usually disagree.

    .venv/bin/python scripts/crosscheck_edges.py --per-kind 2000 --seed 0

Random playouts almost never produce these; this script constructs them.

Kinds
-----
tie        several capture sequences of equal maximal length (incl. same
           captured set via different landing squares, and different sets)
flying     a king with multiple landing squares after a capture, where the
           landing choice changes which continuations exist
backrank   a man whose capture path passes through its promotion rank and
           must keep capturing as a man (no mid-sequence promotion), plus
           the case where it *ends* there and must promote
obstacle   a captured piece stays on the board and blocks a later landing
           square / a later jump in the same sequence (and a cycle around a
           captured piece that must not be jumped twice)
dense      many opponent pieces around a single king: deep branching
mixed      men and kings both able to capture, different lengths
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

from dq.ref_pydraughts import ref_legal_moves
from dq.rules import (
    BACK_RANK, BLACK, NEXT, NSQ, WHITE, Move, Position, bit, capture_sequences,
    legal_moves, squares,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _place(rng: random.Random, *, wm=(), wk=(), bm=(), bk=(), turn=WHITE) -> Position | None:
    m = {"wm": 0, "wk": 0, "bm": 0, "bk": 0}
    used = 0
    for key, sq in (("wm", wm), ("wk", wk), ("bm", bm), ("bk", bk)):
        for s in sq:
            if not 1 <= s <= NSQ or used & bit(s):
                return None
            used |= bit(s)
            m[key] |= bit(s)
    pos = Position(m["wm"], m["wk"], m["bm"], m["bk"], turn)
    try:
        pos.validate()
    except ValueError:
        return None
    return pos


# Opposite of direction d is d ^ 3 (DIRS order: UL, UR, DL, DR).
def _step(i: int, d: int, n: int = 1) -> int:
    """Square index after n steps in direction d, or -1."""
    for _ in range(n):
        i = NEXT[i][d]
        if i < 0:
            return -1
    return i


def _random_filler(rng: random.Random, pos: Position, n: int, colour: int, king_prob: float) -> Position:
    """Drop n extra random pieces of `colour` onto empty squares (never on
    their own promotion rank as men)."""
    wm, wk, bm, bk = pos.wm, pos.wk, pos.bm, pos.bk
    empties = [s for s in range(1, NSQ + 1) if not pos.occupied & bit(s)]
    rng.shuffle(empties)
    for s in empties[:n]:
        king = rng.random() < king_prob or bool(bit(s) & BACK_RANK[colour])
        if colour == WHITE:
            if king: wk |= bit(s)
            else: wm |= bit(s)
        else:
            if king: bk |= bit(s)
            else: bm |= bit(s)
    return Position(wm, wk, bm, bk, pos.turn)


# ---------------------------------------------------------------------------
# generators (each yields candidate Positions; may yield None)
# ---------------------------------------------------------------------------

def gen_tie(rng: random.Random) -> Position | None:
    """Two or more independent capture branches of the same length.
    Build a 'V' or 'X' around a white man/king: enemy pieces in two or more
    directions each with an empty landing square behind."""
    s = rng.randrange(1, NSQ + 1)
    king = rng.random() < 0.5
    dirs = rng.sample(range(4), k=rng.choice([2, 2, 3, 4]))
    black = []
    for d in dirs:
        j = _step(s - 1, d)
        k = _step(s - 1, d, 2)
        if j < 0 or k < 0:
            return None
        black.append(j + 1)
        # sometimes extend the branch so lengths differ by exactly 0 or 1
        if rng.random() < 0.5:
            d2 = rng.choice([x for x in range(4) if x != (d ^ 3)])
            j2 = _step(k, d2)
            k2 = _step(k, d2, 2)
            if j2 >= 0 and k2 >= 0 and (j2 + 1) not in black:
                black.append(j2 + 1)
    pos = _place(rng, wk=[s] if king else [], wm=[] if king else [s], bm=black)
    if pos is None:
        return None
    return _random_filler(rng, pos, rng.randint(0, 3), BLACK, 0.3)


def gen_flying(rng: random.Random) -> Position | None:
    """King far from an enemy piece along a diagonal with several empty
    squares beyond it; put a second enemy piece reachable only from some
    landing squares."""
    s = rng.randrange(1, NSQ + 1)
    d = rng.randrange(4)
    # walk 0-3 empty squares, then the victim, then >=2 empty beyond
    gap = rng.randint(0, 3)
    j = _step(s - 1, d, gap + 1)
    if j < 0:
        return None
    beyond = []
    k = NEXT[j][d]
    while k >= 0:
        beyond.append(k)
        k = NEXT[k][d]
    if len(beyond) < 2:
        return None
    black = [j + 1]
    # second victim placed off one of the landing squares (not the first)
    land = rng.choice(beyond[1:])
    d2 = rng.choice([x for x in range(4) if x != d and x != (d ^ 3)])
    j2 = _step(land, d2)
    k2 = _step(land, d2, 2)
    if j2 >= 0 and k2 >= 0 and j2 not in beyond and j2 != j:
        black.append(j2 + 1)
    pos = _place(rng, wk=[s], bm=black)
    if pos is None:
        return None
    return _random_filler(rng, pos, rng.randint(0, 3), rng.choice([WHITE, BLACK]), 0.4)


def gen_backrank(rng: random.Random) -> Position | None:
    """White man two rows below the back rank; a victim one step forward so
    the landing square is ON the back rank; then a second victim reachable
    from there (backward capture).  Half the time omit the second victim so
    the sequence ends on the back rank and must promote."""
    # white man on rank index 5 (row 2 from top) -> squares 9..12 are rank 7? compute via geometry
    cands = []
    for s in range(1, NSQ + 1):
        for d in (0, 1):  # white forward dirs
            j = _step(s - 1, d)
            k = _step(s - 1, d, 2)
            if j >= 0 and k >= 0 and bit(k + 1) & BACK_RANK[WHITE]:
                cands.append((s, d, j, k))
    s, d, j, k = rng.choice(cands)
    black = [j + 1]
    cont = rng.random() < 0.6
    if cont:
        # backward direction from the landing square (2 or 3), with empty landing
        d2 = rng.choice([2, 3])
        j2 = _step(k, d2)
        k2 = _step(k, d2, 2)
        if j2 < 0 or k2 < 0 or j2 == (s - 1):
            return None
        black.append(j2 + 1)
        # optionally a third victim to make the "stayed a man" vs "became king" distinction bite:
        # a king could fly past, a man cannot.
        if rng.random() < 0.5:
            j3 = _step(k2, d2, 2)   # one empty square gap, then victim: only a king could take it
            k3 = _step(k2, d2, 3)
            if j3 >= 0 and k3 >= 0:
                black.append(j3 + 1)
    pos = _place(rng, wm=[s], bm=black)
    if pos is None:
        return None
    return _random_filler(rng, pos, rng.randint(0, 2), BLACK, 0.2)


def gen_obstacle(rng: random.Random) -> Position | None:
    """A square-shaped cycle: king (or man) can capture around four victims
    arranged in a ring; the first captured piece must still block at the end.
    Also: a victim whose 'behind' square is occupied by an already-captured
    piece's neighbour."""
    s = rng.randrange(1, NSQ + 1)
    i = s - 1
    # ring: go d0 two steps, then d1 two steps, then d0^2 two steps, then d1^2 -> back to start
    d0 = rng.randrange(4)
    d1 = rng.choice([x for x in range(4) if x != d0 and x != (d0 ^ 3)])
    victims = []
    cur = i
    for d in (d0, d1, d0 ^ 3, d1 ^ 3):
        v = _step(cur, d)
        nxt = _step(cur, d, 2)
        if v < 0 or nxt < 0:
            return None
        victims.append(v + 1)
        cur = nxt
    if cur != i or len(set(victims)) != 4:
        return None
    king = rng.random() < 0.6
    pos = _place(rng, wk=[s] if king else [], wm=[] if king else [s], bm=victims)
    if pos is None:
        return None
    return _random_filler(rng, pos, rng.randint(0, 3), rng.choice([WHITE, BLACK]), 0.3)


def gen_dense(rng: random.Random) -> Position | None:
    """A king in the middle with 4-8 enemy pieces within two steps in
    assorted directions; many empty landing squares."""
    s = rng.choice([10, 11, 14, 15, 18, 19, 22, 23, 13, 20])
    cells = set()
    i = s - 1
    for d in range(4):
        a = _step(i, d)
        b = _step(i, d, 2)
        c = _step(i, d, 3)
        for x in (a, b, c):
            if x >= 0:
                cells.add(x + 1)
        if a >= 0:
            for d2 in range(4):
                y = _step(a, d2)
                if y >= 0:
                    cells.add(y + 1)
    cells.discard(s)
    cells = list(cells)
    rng.shuffle(cells)
    black = cells[: rng.randint(4, min(8, len(cells)))]
    pos = _place(rng, wk=[s], bm=[b for b in black if not bit(b) & BACK_RANK[BLACK]],
                 bk=[b for b in black if bit(b) & BACK_RANK[BLACK]])
    return pos


def gen_mixed(rng: random.Random) -> Position | None:
    """Both a white man and a white king can capture; lengths may differ,
    so max-capture must pick across piece types."""
    a = gen_tie(rng)
    if a is None:
        return None
    # add a white king somewhere with a capture available
    for _ in range(10):
        s = rng.randrange(1, NSQ + 1)
        if a.occupied & bit(s):
            continue
        d = rng.randrange(4)
        j = _step(s - 1, d, rng.randint(1, 3))
        k = NEXT[j][d] if j >= 0 else -1
        if j < 0 or k < 0 or a.occupied & (bit(j + 1) | bit(k + 1)):
            continue
        pos = Position(a.wm, a.wk | bit(s), a.bm | bit(j + 1), a.bk, WHITE)
        try:
            pos.validate()
        except ValueError:
            continue
        return pos
    return None


GENS = {
    "tie": gen_tie,
    "flying": gen_flying,
    "backrank": gen_backrank,
    "obstacle": gen_obstacle,
    "dense": gen_dense,
    "mixed": gen_mixed,
}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def compare(pos: Position) -> tuple[bool, list[Move], list[Move]]:
    mine = sorted(legal_moves(pos))
    ref = sorted(ref_legal_moves(pos))
    return mine == ref, mine, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default=",".join(GENS))
    ap.add_argument("--both-colours", action="store_true", default=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    stats: Counter[str] = Counter()
    interesting: Counter[str] = Counter()   # positions where a capture existed
    multi: Counter[str] = Counter()         # positions with >1 legal capture after max filter
    for kind in a.kinds.split(","):
        gen = GENS[kind]
        n = 0
        tries = 0
        while n < a.per_kind and tries < a.per_kind * 50:
            tries += 1
            pos = gen(rng)
            if pos is None:
                continue
            variants = [pos]
            if a.both_colours:
                # mirror: swap colours and rotate so Black has the same shape
                variants.append(_swap(pos))
            for p in variants:
                try:
                    p.validate()
                except ValueError:
                    continue
                ok, mine, ref = compare(p)
                n += 1
                stats[kind] += 1
                caps = capture_sequences(p)
                if caps:
                    interesting[kind] += 1
                    if len(mine) > 1 and mine[0].is_capture:
                        multi[kind] += 1
                if not ok:
                    print(f"MISMATCH kind={kind}\n{p.fen()}\n{p.ascii()}")
                    print("native    :", [str(m) for m in mine])
                    print("pydraughts:", [str(m) for m in ref])
                    print("all native capture seqs:", [str(m) for m in sorted(caps)])
                    return 1
        print(f"{kind:9s} checked={stats[kind]:6d} with_capture={interesting[kind]:6d} "
              f"multi_choice={multi[kind]:6d}  (tries {tries})")
    print(f"OK: {sum(stats.values())} positions agree ({time.time() - t0:.1f}s)")
    return 0


def _rot(mask: int) -> int:
    out = 0
    for s in squares(mask):
        out |= bit(33 - s)
    return out


def _swap(pos: Position) -> Position:
    """Colour-swap + 180 rotation: the same shape from Black's side."""
    return Position(_rot(pos.bm), _rot(pos.bk), _rot(pos.wm), _rot(pos.wk), pos.turn ^ 1)


if __name__ == "__main__":
    sys.exit(main())
