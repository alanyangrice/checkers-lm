"""Expert encoder: a small transformer over the 32 board tokens (PLAN §4).

Heads (all from the mover's point of view, inputs from :func:`dq.tokens.encode_board`):

* ``policy``  — logits over (from, to) square pairs, ``32 * 32 = 1024``
  classes indexed ``(from - 1) * 32 + (to - 1)`` in POV squares.  Illegal
  pairs are masked at use time.  Two legal capture paths sharing the same
  (from, to) are rare; they split the pair's probability equally.
* ``wdl``     — 3 logits: win / draw / loss for the side to move.
* ``ncap``    — aux head: number of pieces captured by the maximal capture
  sequence (0 = quiet position), classes 0..MAX_CAP.  Keeps capture
  structure in the residual stream (PLAN risk 1).

``NetEvaluator`` adapts a trained model to the :mod:`dq.mcts` evaluator
interface.  ``Example`` is the self-play record; ``collate`` turns a list
of them into tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dq.rules import NSQ, Move, Position, legal_moves
from dq.tokens import BOARD_VOCAB, encode_board, pov_move

N_PAIRS = NSQ * NSQ
MAX_CAP = 8          # a capture of >8 pieces is possible in theory, clamp
WDL_W, WDL_D, WDL_L = 0, 1, 2


def pair_index(mv: Move) -> int:
    """Policy index of a POV move."""
    return (mv.start - 1) * NSQ + (mv.end - 1)


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0


class SquareTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.tok = nn.Embedding(BOARD_VOCAB, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, NSQ, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        # policy: bilinear-ish over (from, to) square embeddings
        self.pol_from = nn.Linear(cfg.d_model, cfg.d_model)
        self.pol_to = nn.Linear(cfg.d_model, cfg.d_model)
        self.wdl = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 3))
        self.ncap = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, MAX_CAP + 1))

    def hidden_states(self, ids: torch.Tensor) -> list[torch.Tensor]:
        """Per-layer hidden states (for the LM bridge, PLAN §4).  ``ids`` is
        ``(B, 32)`` long.  Returns ``n_layers + 1`` tensors ``(B, 32, d)``."""
        x = self.tok(ids) + self.pos
        out = [x]
        for blk in self.blocks.layers:
            x = blk(x)
            out.append(x)
        return out

    def forward(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.norm(self.hidden_states(ids)[-1])          # (B, 32, d)
        f = self.pol_from(x); t = self.pol_to(x)
        policy = torch.einsum("bid,bjd->bij", f, t) / (self.cfg.d_model ** 0.5)   # (B, 32, 32)
        pooled = x.mean(dim=1)
        return {
            "policy": policy.reshape(ids.shape[0], N_PAIRS),
            "wdl": self.wdl(pooled),
            "ncap": self.ncap(pooled),
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# training examples
# ---------------------------------------------------------------------------

@dataclass
class Example:
    ids: list[int]                   # 32 board tokens (POV)
    policy: dict[int, float]         # pair index -> target prob (POV), sums to 1
    wdl: int                         # WDL_W / WDL_D / WDL_L for the side to move
    ncap: int                        # max-capture length at this position


def make_example(pos: Position, visit_policy: dict[Move, float], result_for_mover: float) -> Example:
    """``result_for_mover``: +1 win, 0 draw, -1 loss for the side to move at ``pos``."""
    ids = encode_board(pos)
    pol: dict[int, float] = {}
    for mv, p in visit_policy.items():
        k = pair_index(pov_move(mv, pos.turn))
        pol[k] = pol.get(k, 0.0) + p
    moves = legal_moves(pos)
    ncap = max((len(m.captured) for m in moves), default=0)
    wdl = WDL_W if result_for_mover > 0.5 else WDL_L if result_for_mover < -0.5 else WDL_D
    return Example(ids, pol, wdl, min(ncap, MAX_CAP))


def collate(batch: list[Example], device=None) -> dict[str, torch.Tensor]:
    """Dense tensors for a batch.  The policy target is scattered from the
    sparse per-example dicts in one call."""
    B = len(batch)
    ids = torch.tensor([e.ids for e in batch], dtype=torch.long)
    rows, cols, vals = [], [], []
    for i, e in enumerate(batch):
        rows.extend([i] * len(e.policy))
        cols.extend(e.policy.keys())
        vals.extend(e.policy.values())
    pol = torch.zeros(B, N_PAIRS)
    if rows:
        pol.index_put_((torch.tensor(rows), torch.tensor(cols)), torch.tensor(vals, dtype=torch.float32), accumulate=True)
    wdl = torch.tensor([e.wdl for e in batch], dtype=torch.long)
    ncap = torch.tensor([e.ncap for e in batch], dtype=torch.long)
    out = {"ids": ids, "policy": pol, "wdl": wdl, "ncap": ncap}
    if device is not None:
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
            w_pol: float = 1.0, w_wdl: float = 1.0, w_cap: float = 0.25) -> tuple[torch.Tensor, dict[str, float]]:
    # policy: cross-entropy against a soft target; mask pairs with zero target
    # *and* zero legal mass is handled by the target itself (illegal pairs have 0)
    logp = F.log_softmax(out["policy"], dim=-1)
    l_pol = -(batch["policy"] * logp).sum(-1).mean()
    l_wdl = F.cross_entropy(out["wdl"], batch["wdl"])
    l_cap = F.cross_entropy(out["ncap"], batch["ncap"])
    total = w_pol * l_pol + w_wdl * l_wdl + w_cap * l_cap
    return total, {"policy": l_pol.item(), "wdl": l_wdl.item(), "ncap": l_cap.item()}


# ---------------------------------------------------------------------------
# MCTS adapter
# ---------------------------------------------------------------------------

class NetEvaluator:
    """Batched evaluator for :class:`dq.mcts.MCTS`: one forward pass per call.

    Values are ``P(win) - P(loss)`` from the WDL head.  Priors are the
    softmax over the legal (from, to) pairs; moves sharing a pair split its
    mass equally."""

    def __init__(self, model: SquareTransformer, device=None, autocast: bool = False,
                 contempt: float = 0.0) -> None:
        """``contempt``: Lc0-style WDL draw contempt (v0.30+).  The value handed to the search
        is ``P(win) - P(loss) - c * P(draw)`` when the leaf's side to move is the side that owns
        the search (the root player), and ``+ c * P(draw)`` otherwise, so the backup stays
        zero-sum: the root player avoids draws, its opponent is assumed to want them.  Labels
        are never touched; this only steers play.  Requires the caller to pass ``root_turns``."""
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.autocast = autocast and self.device.type == "cuda"
        self.contempt = contempt
        self.calls = 0
        self.positions = 0

    @torch.no_grad()
    def __call__(self, positions: list[Position], moves: list[list[Move]],
                 root_turns: list[int] | None = None) -> list[tuple[dict[Move, float], float]]:
        B = len(positions)
        self.calls += 1
        self.positions += B
        ids = torch.tensor([encode_board(p) for p in positions], dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            out = self.model(ids)
        logits = out["policy"].float()
        wdl = torch.softmax(out["wdl"].float(), dim=-1)
        v = wdl[:, WDL_W] - wdl[:, WDL_L]
        if self.contempt and root_turns is not None:
            sign = torch.tensor([1.0 if p.turn == rt else -1.0 for p, rt in zip(positions, root_turns)],
                                device=v.device)
            v = v - self.contempt * wdl[:, WDL_D] * sign
        values = v.tolist()
        # gather legal-pair logits with a padded index matrix and one softmax
        idx_lists = [[pair_index(pov_move(m, p.turn)) for m in ms] for p, ms in zip(positions, moves)]
        L = max(len(x) for x in idx_lists)
        idx = torch.zeros(B, L, dtype=torch.long)
        mask = torch.full((B, L), float("-inf"))
        for i, x in enumerate(idx_lists):
            idx[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 0.0
        idx = idx.to(self.device); mask = mask.to(self.device)
        sel = logits.gather(1, idx) + mask
        probs = torch.softmax(sel, dim=-1).cpu().tolist()
        out_list = []
        for ms, x, pr in zip(moves, idx_lists, probs):
            counts: dict[int, int] = {}
            for k in x:
                counts[k] = counts.get(k, 0) + 1
            prior = {m: pr[j] / counts[k] for j, (m, k) in enumerate(zip(ms, x))}
            z = sum(prior.values()) or 1.0
            out_list.append(({m: p / z for m, p in prior.items()}, values[len(out_list)]))
        return out_list
