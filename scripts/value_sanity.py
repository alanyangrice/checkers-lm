"""Value-head sanity and calibration checks for a checkpoint.

    python scripts/value_sanity.py runs/gen4/weights/latest.pt --tb data/tb

Sections:
  1. Tablebase positions by true class (WIN / DRAW / LOSS, draw-rule adjusted): mean predicted
     P(win), P(draw), P(loss), argmax accuracy, and mean value V = P(win) - P(loss).
  2. Calibration: bucket positions by predicted V and compare with the true mean outcome.
  3. Hand-picked absolutes: start position (should be near draw), huge material edges beyond
     tablebase range (should be near +1 / -1), and random 5-8 piece positions (spread).
  4. Symmetry: a position and its colour-flipped mirror must get the same value.
"""
from __future__ import annotations

import argparse
import random

import torch

from dq.game import GameState, draw_ply_limit
from dq.model import WDL_D, WDL_L, WDL_W, ModelConfig, SquareTransformer
from dq.rules import BLACK, WHITE, Position, bit, legal_moves, random_position
from dq.selfplay import tb_outcome
from dq.tablebase import DRAW, LOSS, WIN, Tablebase
from dq.tokens import encode_board


@torch.no_grad()
def wdl_probs(model, positions, device):
    ids = torch.tensor([encode_board(p) for p in positions], dtype=torch.long, device=device)
    out = []
    for i in range(0, len(ids), 2048):
        out.append(torch.softmax(model(ids[i:i + 2048])["wdl"].float(), -1).cpu())
    return torch.cat(out)


def _rot(mask):
    out = 0
    for s in range(1, 33):
        if mask & bit(s):
            out |= bit(33 - s)
    return out


def mirror(p: Position) -> Position:
    """Same position seen from the other colour (rotate + swap): identical POV encoding."""
    return Position(_rot(p.bm), _rot(p.bk), _rot(p.wm), _rot(p.wk), WHITE if p.turn == BLACK else BLACK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tb", default="data/tb")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SquareTransformer(ModelConfig()).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev)); model.eval()
    tb = Tablebase.load(a.tb)
    rng = random.Random(a.seed)

    # ---- 1. tablebase classes --------------------------------------------------------
    by_cls = {WIN: [], DRAW: [], LOSS: []}
    tries = 0
    while min(len(v) for v in by_cls.values()) < a.n // 3 and tries < 200_000:
        tries += 1
        n_w, n_b = rng.randint(1, 3), rng.randint(1, 3)
        if n_w + n_b > 4:
            continue
        p = random_position(rng, n_w, n_b, king_prob=0.5)
        if not tb.covers(p) or not legal_moves(p):
            continue
        v = tb_outcome(tb, p)
        if len(by_cls[v]) < a.n // 3:
            by_cls[v].append(p)
    print(f"== 1. tablebase positions ({', '.join(f'{k}:{len(v)}' for k, v in by_cls.items())}; WIN=1 DRAW=0 LOSS=-1)")
    print(f"{'true':>6} {'P(win)':>7} {'P(draw)':>8} {'P(loss)':>8} {'mean V':>7} {'argmax acc':>11}")
    all_pos, all_true = [], []
    for cls, name in ((WIN, "WIN"), (DRAW, "DRAW"), (LOSS, "LOSS")):
        ps = by_cls[cls]
        pr = wdl_probs(model, ps, dev)
        target = {WIN: WDL_W, DRAW: WDL_D, LOSS: WDL_L}[cls]
        acc = (pr.argmax(-1) == target).float().mean().item()
        V = (pr[:, WDL_W] - pr[:, WDL_L]).mean().item()
        print(f"{name:>6} {pr[:, WDL_W].mean():7.3f} {pr[:, WDL_D].mean():8.3f} {pr[:, WDL_L].mean():8.3f} {V:7.3f} {acc:11.3f}")
        all_pos += ps; all_true += [float(cls)] * len(ps)

    # ---- 2. calibration ---------------------------------------------------------------
    pr = wdl_probs(model, all_pos, dev)
    V = pr[:, WDL_W] - pr[:, WDL_L]
    true = torch.tensor(all_true)
    print("\n== 2. calibration on the same positions: predicted V bucket -> true mean outcome")
    edges = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        m = (V >= lo) & (V < hi)
        if m.sum() == 0:
            continue
        print(f"  V in [{lo:5.2f},{hi:5.2f}): n={int(m.sum()):5d}  pred mean V={V[m].mean():6.3f}  true mean={true[m].mean():6.3f}")

    # ---- 3. absolutes ------------------------------------------------------------------
    print("\n== 3. hand-picked positions (V = P(win) - P(loss) for the side to move)")
    def show(label, p):
        q = wdl_probs(model, [p], dev)[0]
        print(f"  {label:<52} P(w/d/l)={q[WDL_W]:.3f}/{q[WDL_D]:.3f}/{q[WDL_L]:.3f}  V={q[WDL_W]-q[WDL_L]:+.3f}")
    show("start position (expect ~draw, V ~ 0)", Position.start())
    # 4 white kings vs 1 black man, white to move: crushing win (5 pieces, beyond tablebase)
    show("4 WK vs 1 BM, white to move (expect V ~ +1)", Position(0, bit(14) | bit(18) | bit(23) | bit(27), bit(10), 0, WHITE))
    show("4 WK vs 1 BM, BLACK to move (expect V ~ -1)", Position(0, bit(14) | bit(18) | bit(23) | bit(27), bit(10), 0, BLACK))
    show("3 WK vs 2 BM, white to move (expect V >> 0)", Position(0, bit(14) | bit(18) | bit(27), bit(9) | bit(10), 0, WHITE))
    show("6 WM vs 6 BM symmetric-ish midgame, white to move", Position(0b111111 << 20, 0, 0b111111, 0, WHITE))
    # wins/losses with a forced capture available: 1-ply tactics
    rnd = [random_position(rng, rng.randint(2, 4), rng.randint(2, 4), 0.3) for _ in range(2000)]
    rnd = [p for p in rnd if legal_moves(p) and not GameState(p).over][:1000]
    q = wdl_probs(model, rnd, dev)
    Vr = q[:, WDL_W] - q[:, WDL_L]
    print(f"  random 4-8 piece positions (n={len(rnd)}): mean V={Vr.mean():+.3f}  std={Vr.std():.3f}  "
          f"|V|>0.8: {(Vr.abs() > 0.8).float().mean():.1%}  |V|<0.2: {(Vr.abs() < 0.2).float().mean():.1%}")

    # ---- 4. symmetry -------------------------------------------------------------------
    print("\n== 4. colour-flip symmetry (same POV encoding => must be identical)")
    mir = [mirror(p) for p in rnd[:500]]
    qa, qb = wdl_probs(model, rnd[:500], dev), wdl_probs(model, mir, dev)
    print(f"  max |P diff| over 500 positions: {(qa - qb).abs().max():.2e} (should be ~0)")


if __name__ == "__main__":
    main()
