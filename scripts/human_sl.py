"""Supervised learning from human (lidraughts) games — AlphaGo's SL policy stage / Maia-style.

Build a dataset of (position, human move, game result) from PDN files and fit the encoder's
policy head to the human move and its WDL head to the game result.  The checkpoint is a
normal state_dict usable with --resume in scripts/selfplay.py.

    python scripts/human_sl.py build  data/lidraughts_brazilian.pdn data/lidraughts_brazilian_top.pdn --out data/human.pkl --min-rating 1800
    python scripts/human_sl.py train  data/human.pkl --out runs/sl_5m.pt --epochs 4 [--init runs/gen8_final.pt] [--d-model 384 --n-layers 8 --d-ff 1536]
    python scripts/human_sl.py eval   runs/sl_5m.pt data/human.pkl      # top-1 move accuracy + WDL accuracy on the held-out split

Positions are stored as Example records (POV tokens, one-hot policy target, WDL label, ncap).
Captures are mandatory in draughts, so plies with a single legal move carry no policy signal
and are skipped for the policy loss (kept for value).  The last 10% of *games* is held out.
"""
from __future__ import annotations

import argparse
import math
import pickle
import random
import re
import sys
import time

import torch

from dq.model import (WDL_D, WDL_L, WDL_W, Example, ModelConfig, SquareTransformer, collate, infer_config,
                      load_model, loss_fn, make_example)
from dq.pdn import read_pdn, replay
from dq.rules import BLACK, WHITE, legal_moves


def build(args) -> None:
    ex: list[tuple[Example, bool]] = []     # (example, has_policy_signal)
    n_games = n_skip = 0
    games_all = []
    for path in args.pdn:
        txt = open(path, encoding="utf-8").read()
        games_all += read_pdn(txt)
    print(f"{len(games_all)} games read", flush=True)
    random.Random(0).shuffle(games_all)
    per_game: list[list[tuple[Example, bool]]] = []
    for g in games_all:
        elos = [int(v) for k, v in g.headers.items() if k in ("WhiteElo", "BlackElo") and str(v).isdigit()]
        if args.min_rating and (len(elos) < 2 or min(elos) < args.min_rating):
            n_skip += 1; continue
        if g.result is None:
            n_skip += 1; continue
        rep = replay(g, keep=True)
        if not rep.ok:
            n_skip += 1; continue
        z_white = 1.0 if g.result == WHITE else -1.0 if g.result == BLACK else 0.0
        rows = []
        for pos, mv in zip(rep.positions, rep.played):
            ms = legal_moves(pos)
            z = z_white if pos.turn == WHITE else -z_white
            rows.append((make_example(pos, {mv: 1.0}, z), len(ms) > 1))
        per_game.append(rows); n_games += 1
    n_hold = max(1, len(per_game) // 10)
    train = [r for rows in per_game[:-n_hold] for r in rows]
    hold = [r for rows in per_game[-n_hold:] for r in rows]
    with open(args.out, "wb") as f:
        pickle.dump({"train": train, "hold": hold}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{n_games} games used ({n_skip} skipped) -> {len(train):,} train / {len(hold):,} held-out positions; "
          f"policy-informative: {sum(p for _, p in train) / max(len(train), 1):.0%}")


def _batches(rows, bs, rng):
    idx = list(range(len(rows))); rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [rows[j] for j in idx[i:i + bs]]


def _loss(model, batch_rows, device):
    exs = [e for e, _ in batch_rows]
    has_pol = torch.tensor([p for _, p in batch_rows], dtype=torch.float32, device=device)
    b = collate(exs, device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(b["ids"])
    out = {k: v.float() for k, v in out.items()}
    logp = torch.log_softmax(out["policy"], -1)
    l_pol = -(b["policy"] * logp).sum(-1)
    l_pol = (l_pol * has_pol).sum() / has_pol.sum().clamp(min=1)
    l_wdl = torch.nn.functional.cross_entropy(out["wdl"], b["wdl"])
    l_cap = torch.nn.functional.cross_entropy(out["ncap"], b["ncap"])
    return l_pol + l_wdl + 0.25 * l_cap, l_pol.item(), l_wdl.item(), out, b, has_pol


@torch.no_grad()
def evaluate(model, rows, device, bs=2048):
    model.eval(); top1 = n_pol = wdl_ok = n = 0
    for batch in _batches(rows, bs, random.Random(0)):
        _, _, _, out, b, has_pol = _loss(model, batch, device)
        # top-1 among *legal* moves: legal pairs are exactly the keys with any target mass? no - target is one-hot.
        # Use argmax over all pairs vs the human pair; illegal argmax counts as wrong (strict).
        pred = out["policy"].argmax(-1); truth = b["policy"].argmax(-1)
        m = has_pol > 0
        top1 += ((pred == truth) & m).sum().item(); n_pol += m.sum().item()
        wdl_ok += (out["wdl"].argmax(-1) == b["wdl"]).sum().item(); n += len(batch)
    return top1 / max(n_pol, 1), wdl_ok / max(n, 1)


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(args.data, "rb"))
    tr, ho = data["train"], data["hold"]
    if args.init:
        model = load_model(args.init, device)
        print(f"init from {args.init}: {model.n_params():,} params")
    else:
        model = SquareTransformer(ModelConfig(d_model=args.d_model, n_layers=args.n_layers,
                                              n_heads=args.n_heads, d_ff=args.d_ff)).to(device)
        print(f"fresh model: {model.n_params():,} params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_total = args.epochs * math.ceil(len(tr) / args.batch); step = 0
    rng = random.Random(1)
    acc, wacc = evaluate(model, ho, device)
    print(f"before: held-out top-1 move acc {acc:.3f}, WDL acc {wacc:.3f}", flush=True)
    for ep in range(args.epochs):
        model.train(); t = time.time(); sp = sw = 0.0; nb = 0
        for batch in _batches(tr, args.batch, rng):
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps_total)); step += 1
            for pg in opt.param_groups: pg["lr"] = lr
            loss, lp, lw, *_ = _loss(model, batch, device)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            sp += lp; sw += lw; nb += 1
        acc, wacc = evaluate(model, ho, device)
        print(f"epoch {ep+1}/{args.epochs}: policy {sp/nb:.3f} wdl {sw/nb:.3f} | held-out top-1 {acc:.3f} WDL acc {wacc:.3f} "
              f"({time.time()-t:.0f}s)", flush=True)
        torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


def eval_cmd(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = pickle.load(open(args.data, "rb"))
    model = load_model(args.ckpt, device)
    acc, wacc = evaluate(model, data["hold"], device)
    print(f"{args.ckpt}: held-out top-1 move acc {acc:.3f}, WDL acc {wacc:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("pdn", nargs="+"); b.add_argument("--out", default="data/human.pkl")
    b.add_argument("--min-rating", type=int, default=0)
    t = sub.add_parser("train"); t.add_argument("data"); t.add_argument("--out", required=True); t.add_argument("--init")
    t.add_argument("--epochs", type=int, default=4); t.add_argument("--batch", type=int, default=512); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--d-model", type=int, default=256); t.add_argument("--n-layers", type=int, default=6)
    t.add_argument("--n-heads", type=int, default=8); t.add_argument("--d-ff", type=int, default=1024)
    e = sub.add_parser("eval"); e.add_argument("ckpt"); e.add_argument("data")
    a = ap.parse_args()
    {"build": build, "train": train, "eval": eval_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
