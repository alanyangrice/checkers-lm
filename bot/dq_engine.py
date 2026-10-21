"""Homemade engine for lidraughts-bot (https://github.com/AttackingOrDefending/lidraughts-bot).

Setup
-----
1. ``git clone https://github.com/AttackingOrDefending/lidraughts-bot`` next to
   this repo and ``pip install -r lidraughts-bot/requirements.txt`` into the
   same venv that has ``draughts-queen`` installed (``uv pip install -e .``).
2. Copy ``bot/config.yml`` to ``lidraughts-bot/config.yml`` and put your BOT
   account token in it (create a token with the ``bot:play`` scope; the
   account must be upgraded to a BOT account once, see the lidraughts-bot
   README).
3. ``cp bot/dq_engine.py lidraughts-bot/`` and in ``lidraughts-bot/strategies.py``
   add ``from dq_engine import DQEngine``.
4. ``cd lidraughts-bot && python lidraughts-bot.py -v``

Environment variables
---------------------
``DQ_WEIGHTS``   path to the encoder checkpoint (default ``runs/gen1/weights/latest.pt``)
``DQ_SIMS``      MCTS simulations per move (default 400)
``DQ_TB``        tablebase directory (default ``data/tb``; empty to disable)
``DQ_DEVICE``    torch device (default: cuda > mps > cpu)

The board lidraughts-bot hands to ``search`` is a ``draughts.Game``; we read
its lidraughts FEN, search with :mod:`dq.mcts`, and hand back the chosen
move as a list of ``[from, to]`` steps.
"""
from __future__ import annotations

import os
import random
import time

import torch

from dq.mcts import MCTS, SearchConfig
from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.rules import Move, Position, apply_move, legal_moves
from dq.tablebase import DRAW, LOSS, Tablebase

try:
    import draughts
    from draughts.engine import PlayResult
    from strategies import MinimalEngine
except ImportError:  # imported outside lidraughts-bot (tests)
    draughts = None

    class MinimalEngine:  # type: ignore[no-redef]
        def __init__(self, *a, **k): ...


def _device() -> torch.device:
    d = os.environ.get("DQ_DEVICE")
    if d:
        return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class DQSearcher:
    """Model + MCTS, independent of lidraughts-bot (used by the engine and tests)."""

    def __init__(self, weights: str | None = None, sims: int | None = None, tb_dir: str | None = None,
                 device: torch.device | None = None, mcfg: ModelConfig | None = None) -> None:
        self.device = device or _device()
        if weights is None:                                      # "" means: no weights, random init
            weights = os.environ.get("DQ_WEIGHTS", "runs/gen1/weights/latest.pt")
        if weights and os.path.exists(weights) and mcfg is None:
            from dq.model import load_model
            self.model = load_model(weights, self.device)       # size inferred from the checkpoint
        else:
            self.model = SquareTransformer(mcfg or ModelConfig()).to(self.device)
            if weights and os.path.exists(weights):
                self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()
        tb_dir = os.environ.get("DQ_TB", "data/tb") if tb_dir is None else tb_dir
        self.tb = Tablebase.load(tb_dir) if tb_dir and os.path.isdir(tb_dir) else None
        self.sims = sims or int(os.environ.get("DQ_SIMS", "400"))
        self.mcts = MCTS(NetEvaluator(self.model, self.device),
                         SearchConfig(simulations=self.sims, dirichlet_eps=0.0, leaves_per_tree=8),
                         tablebase=self.tb, rng=random.Random())

    def tablebase_move(self, pos: Position, moves: list[Move]) -> Move | None:
        """Exact move when the position is covered: win fastest, else draw,
        else lose slowest."""
        if self.tb is None or not self.tb.covers(pos):
            return None
        best, best_key = None, None
        for m in moves:
            c = apply_move(pos, m)
            if c.pieces(c.turn) == 0:
                key = (2, 0)                      # captured their last piece
            else:
                v, d = self.tb.probe(c)           # from the opponent's view
                key = (2, -d) if v == LOSS else (1, 0) if v == DRAW else (0, d)
            if best_key is None or key > best_key:
                best, best_key = m, key
        return best

    def choose(self, pos: Position) -> Move:
        moves = legal_moves(pos)
        if len(moves) == 1:
            return moves[0]
        mv = self.tablebase_move(pos, moves)
        if mv is not None:
            return mv
        root = self.mcts.search(pos, add_noise=False)
        return self.mcts.select_move(root, temperature=0.0)


def to_board_move(mv: Move) -> list[list[int]]:
    return [[a, b] for a, b in zip(mv.path, mv.path[1:])]


class DQEngine(MinimalEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.searcher = DQSearcher()

    def search(self, board, time_limit, ponder, draw_offered):
        pos = Position.from_fen(board.get_li_fen())
        t0 = time.time()
        mv = self.searcher.choose(pos)
        dt = time.time() - t0
        return PlayResult(draughts.Move(board_move=to_board_move(mv)), None,
                          {"string": f"dq {mv} sims={self.searcher.sims} {dt:.2f}s"})
