"""Where does model A lose to model B?  Plays A vs B with search and classifies the decisive
moment of every A loss (and A win, for comparison).

    python scripts/diagnose.py runs/gen9_final.pt runs/gen8_final.pt --games 60 --sims 100 --tb data/tb

For each game we record, per ply, the mover's root Q after search.  In a decisive game the
"blunder ply" is the last ply at which the eventual loser's own search still rated the position
>= -0.3 (it did not see the loss coming) before the evaluation collapsed.  We report, for A's
losses vs A's wins:
  * game phase at the blunder (pieces on board, ply number)
  * whether the loser's chosen move was a capture, a king move, a man move, a promotion
  * whether the opponent's reply was a capture (a tactical shot) and how many pieces it took
  * what the *other* model's raw value said at the blunder position (did B see it?)
  * ending reason (no moves / which draw rule / max plies)
and dumps the blunder positions as FEN to runs/diag_<A>_<B>.jsonl for inspection.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from dq.game import GameState, draw_ply_limit
from dq.mcts import MCTS, Node, SearchConfig
from dq.model import NetEvaluator, load_model
from dq.rules import BLACK, WHITE, Position, legal_moves
from dq.tablebase import Tablebase


def play(ev_a, ev_b, a_white: bool, sims: int, tb, rng, max_plies=300, opening_plies=6):
    sc = SearchConfig(simulations=sims, dirichlet_eps=0.0, leaves_per_tree=8)
    m = {True: MCTS(ev_a, sc, tablebase=tb, rng=rng, tb_ply_limit=draw_ply_limit),
         False: MCTS(ev_b, sc, tablebase=tb, rng=rng, tb_ply_limit=draw_ply_limit)}
    g = GameState(Position.start())
    for _ in range(opening_plies):
        if g.over: break
        g.push(rng.choice(g.legal()))
    trace = []    # (ply, pos, mover_is_a, q_mover, move, raw_value_a, raw_value_b)
    while not g.over and len(g.history) < max_plies:
        pos = g.pos
        mover_is_a = (pos.turn == WHITE) == a_white
        root = Node(pos)
        m[mover_is_a].search_batch([root], add_noise=False, seen=[g.seen])
        mv = m[mover_is_a].select_move(root, 0.0)
        ms = legal_moves(pos)
        (_, va), = ev_a([pos], [ms]); (_, vb), = ev_b([pos], [ms])
        trace.append((len(g.history), pos, mover_is_a, root.Q, mv, va, vb))
        g.push(mv)
    if not g.over:
        g.result, g.reason = 0.5, "max plies"
    return g, trace


def pieces(p: Position) -> int:
    return bin(p.wm | p.wk | p.bm | p.bk).count("1")


def classify(g: GameState, trace, a_white: bool, tb):
    """Return a dict describing the decisive moment, or None for draws."""
    if g.result == 0.5:
        return None
    a_won = (g.result == 1.0) == a_white
    loser_is_a = not a_won
    # plies where the loser moved, with the loser's own root Q
    loser_plies = [t for t in trace if t[2] == loser_is_a]
    if not loser_plies:
        return None
    # last ply at which the loser still thought it was OK
    ok = [t for t in loser_plies if t[3] >= -0.3]
    if not ok:
        bl = loser_plies[0]
    else:
        bl = ok[-1]
    ply, pos, _, q, mv, va, vb = bl
    nxt = [t for t in trace if t[0] == ply + 1]
    reply = nxt[0][4] if nxt else None
    colour = pos.turn
    was_man = bool(pos.men(colour) & (1 << (mv.start - 1)))
    promoted = was_man and bool((1 << (mv.end - 1)) & (0b1111 if colour == WHITE else 0b1111 << 28))
    loser_raw, winner_raw = (va, vb) if loser_is_a else (vb, va)
    return {
        "a_won": a_won, "ply": ply, "pieces": pieces(pos), "fen": pos.fen(),
        "loser_q": round(q, 3), "loser_raw_v": round(loser_raw, 3), "winner_raw_v": round(winner_raw, 3),
        "move": str(mv), "move_type": ("capture" if mv.is_capture else "man" if was_man else "king") + ("+promo" if promoted else ""),
        "reply_capture": bool(reply and reply.is_capture), "reply_captured": len(reply.captured) if reply else 0,
        "tb_covered": bool(tb and tb.covers(pos)), "reason": g.reason, "game_len": len(g.history),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--tb", default="data/tb")
    ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb = Tablebase.load(a.tb)
    ev_a, ev_b = NetEvaluator(load_model(a.a, dev), dev), NetEvaluator(load_model(a.b, dev), dev)
    rng = random.Random(a.seed)
    out = Path("runs") / f"diag_{Path(a.a).stem}_vs_{Path(a.b).stem}.jsonl"
    results = Counter(); reasons = Counter(); recs = []
    with open(out, "w") as f:
        for i in range(a.games):
            g, trace = play(ev_a, ev_b, a_white=(i % 2 == 0), sims=a.sims, tb=tb, rng=rng)
            a_score = g.result if (i % 2 == 0) else 1 - g.result
            results["A win" if a_score == 1 else "draw" if a_score == 0.5 else "A loss"] += 1
            reasons[g.reason] += 1
            c = classify(g, trace, i % 2 == 0, tb)
            if c:
                recs.append(c); f.write(json.dumps(c) + "\n")
            print(f"game {i+1:3d}: {'A win ' if a_score==1 else 'draw  ' if a_score==0.5 else 'A loss'} {g.reason:<16} len={len(g.history)}", flush=True)
    print("\nresults:", dict(results), " endings:", dict(reasons))
    for label, key in (("A LOSSES", False), ("A WINS", True)):
        rs = [r for r in recs if r["a_won"] == key]
        if not rs: continue
        n = len(rs)
        print(f"\n== {label} (n={n}) — decisive moment = last ply the loser still rated >= -0.3")
        ph = Counter("opening(<=20)" if r["ply"] <= 20 else "middle(21-60)" if r["ply"] <= 60 else "late(>60)" for r in rs)
        pc = Counter(">=16 pieces" if r["pieces"] >= 16 else "9-15 pieces" if r["pieces"] >= 9 else "5-8 pieces" if r["pieces"] >= 5 else "<=4 (tablebase)" for r in rs)
        print("  by ply:   ", {k: f"{v/n:.0%}" for k, v in ph.most_common()})
        print("  by pieces:", {k: f"{v/n:.0%}" for k, v in pc.most_common()})
        print("  loser's move type:", {k: f"{v/n:.0%}" for k, v in Counter(r["move_type"] for r in rs).most_common()})
        shots = [r for r in rs if r["reply_capture"]]
        print(f"  opponent's reply was a capture (shot): {len(shots)/n:.0%}; multi-piece: {sum(r['reply_captured']>=2 for r in rs)/n:.0%}")
        print(f"  mean loser root Q at blunder {sum(r['loser_q'] for r in rs)/n:+.2f}; loser raw V {sum(r['loser_raw_v'] for r in rs)/n:+.2f}; "
              f"winner raw V (same position, mover POV) {sum(r['winner_raw_v'] for r in rs)/n:+.2f}")
        print(f"  winner's net already saw it (raw V <= -0.3 for the mover): {sum(r['winner_raw_v'] <= -0.3 for r in rs)/n:.0%}; "
              f"loser's net saw it too: {sum(r['loser_raw_v'] <= -0.3 for r in rs)/n:.0%}")
    print(f"\nblunder positions written to {out}")


if __name__ == "__main__":
    main()
