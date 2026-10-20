"""PUCT Monte-Carlo tree search over :mod:`dq.rules` (AlphaZero-style).

Two entry points share one tree implementation:

* :meth:`MCTS.search` — one position, one leaf per simulation.  Simple,
  used by tests and by evaluation matches.
* :meth:`MCTS.search_batch` — many independent trees advanced in lockstep.
  Each round collects up to ``leaves_per_tree`` leaves from every tree
  using virtual loss, evaluates them all in **one** evaluator call, then
  backs up.  With 32 trees × 8 leaves that is a 256-position forward pass
  per round with no inter-process traffic, which is how self-play keeps a
  GPU busy from a handful of Python processes.

Evaluator interface (batched; a single-position call is the batch of one)::

    evaluator(positions: list[Position], moves: list[list[Move]])
        -> list[tuple[dict[Move, float], float]]

``value`` is in [-1, 1] from the point of view of the side to move.

Terminal handling: a position with no legal move is a loss (-1) for the
side to move.  If a :class:`dq.tablebase.Tablebase` is supplied, covered
interior positions are scored exactly and not expanded; the root is always
expanded because a move must be chosen there.

Trees can be reused between moves with :meth:`MCTS.advance`.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from dq.rules import Move, Position, apply_move, legal_moves

BatchEvaluator = Callable[[list[Position], list[list[Move]]], list[tuple[dict[Move, float], float]]]


def uniform_evaluator(positions: list[Position], moves: list[list[Move]]) -> list[tuple[dict[Move, float], float]]:
    return [({m: 1.0 / len(ms) for m in ms}, 0.0) for ms in moves]


@dataclass
class Node:
    pos: Position
    moves: list[Move] = field(default_factory=list)
    prior: dict[Move, float] = field(default_factory=dict)
    children: dict[Move, "Node"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0            # sum of values, perspective of the side to move here
    vloss: int = 0            # virtual-loss count while a batch is in flight
    terminal_value: Optional[float] = None

    @property
    def Q(self) -> float:
        n = self.N + self.vloss
        return (self.W - self.vloss) / n if n else 0.0   # virtual loss counts as -1

    @property
    def expanded(self) -> bool:
        return bool(self.prior) or self.terminal_value is not None


@dataclass
class SearchConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25        # 0 disables root noise
    fpu: float = 0.0                   # first-play-urgency value for unvisited children
    leaves_per_tree: int = 8           # search_batch only


class MCTS:
    def __init__(self, evaluator: BatchEvaluator, config: SearchConfig | None = None,
                 tablebase=None, rng: random.Random | None = None) -> None:
        self.eval = evaluator
        self.cfg = config or SearchConfig()
        self.tb = tablebase
        self.rng = rng or random.Random()

    # -- node preparation ---------------------------------------------------
    def _prepare(self, node: Node, use_tb: bool = True) -> Optional[float]:
        """Fill ``moves`` and decide whether the node is terminal / tablebase.
        Returns the exact value if so, else None (needs the evaluator)."""
        if node.terminal_value is not None:
            return node.terminal_value
        if not node.moves:
            node.moves = legal_moves(node.pos)
        if not node.moves:
            node.terminal_value = -1.0
            return -1.0
        if use_tb and self.tb is not None and self.tb.covers(node.pos):
            r = self.tb.probe(node.pos)
            if r is not None:
                node.terminal_value = float(r[0])
                return node.terminal_value
        return None

    def _expand_root(self, root: Node, add_noise: bool) -> None:
        if root.expanded and root.terminal_value is None:
            if add_noise:
                self._add_root_noise(root)
            return
        root.terminal_value = None
        if self._prepare(root, use_tb=False) is None:
            (prior, _), = self.eval([root.pos], [root.moves])
            root.prior = prior
        if add_noise:
            self._add_root_noise(root)

    def _add_root_noise(self, root: Node) -> None:
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        if eps <= 0 or len(root.moves) <= 1 or not root.prior:
            return
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in root.moves]
        s = sum(noise)
        for m, n in zip(root.moves, noise):
            root.prior[m] = (1 - eps) * root.prior[m] + eps * n / s

    # -- selection ------------------------------------------------------------
    def _select(self, node: Node) -> Move:
        sqrt_n = math.sqrt(node.N + node.vloss + 1)
        c = self.cfg.c_puct
        best, best_score = None, -1e9
        for m in node.moves:
            child = node.children.get(m)
            if child is not None and (child.N or child.vloss):
                q = -child.Q
                u = c * node.prior[m] * sqrt_n / (1 + child.N + child.vloss)
            else:
                q = self.cfg.fpu
                u = c * node.prior[m] * sqrt_n
            score = q + u
            if score > best_score:
                best, best_score = m, score
        return best  # type: ignore[return-value]

    def _descend(self, root: Node) -> list[Node]:
        """Walk from root to a leaf (unexpanded or terminal), applying virtual
        loss along the way.  Returns the path."""
        node = root
        path = [root]
        while node.expanded and node.terminal_value is None:
            m = self._select(node)
            child = node.children.get(m)
            if child is None:
                child = Node(apply_move(node.pos, m))
                node.children[m] = child
            node = child
            path.append(node)
        for n in path:
            n.vloss += 1
        return path

    @staticmethod
    def _backup(path: list[Node], value: float) -> None:
        """``value`` is from the leaf mover's perspective."""
        for n in reversed(path):
            n.vloss -= 1
            n.N += 1
            n.W += value
            value = -value

    # -- single-tree search -------------------------------------------------
    def search(self, pos: Position, root: Node | None = None, add_noise: bool = True) -> Node:
        root = root or Node(pos)
        self.search_batch([root], add_noise=add_noise, leaves_per_tree=1)
        return root

    # -- lockstep batched search ---------------------------------------------
    def search_batch(self, roots: list[Node], add_noise: bool = True,
                     leaves_per_tree: int | None = None) -> None:
        lpt = leaves_per_tree or self.cfg.leaves_per_tree
        for r in roots:
            self._expand_root(r, add_noise)
        remaining = [self.cfg.simulations] * len(roots)
        while any(remaining):
            pending_paths: list[list[Node]] = []
            pending_idx: list[int] = []
            for i, root in enumerate(roots):
                k = min(lpt, remaining[i])
                for _ in range(k):
                    path = self._descend(root)
                    leaf = path[-1]
                    v = self._prepare(leaf)
                    if v is not None:
                        self._backup(path, v)
                        remaining[i] -= 1
                    else:
                        pending_paths.append(path)
                        pending_idx.append(i)
            if not pending_paths:
                continue
            # evaluate each distinct leaf once (two paths may reach the same leaf)
            uniq: dict[int, Node] = {}
            for path in pending_paths:
                uniq.setdefault(id(path[-1]), path[-1])
            leaves = list(uniq.values())
            results = self.eval([n.pos for n in leaves], [n.moves for n in leaves])
            val_by_id = {}
            for n, (prior, value) in zip(leaves, results):
                n.prior = prior
                val_by_id[id(n)] = value
            for path, i in zip(pending_paths, pending_idx):
                self._backup(path, val_by_id[id(path[-1])])
                remaining[i] -= 1

    # -- results ------------------------------------------------------------
    def visit_counts(self, root: Node) -> dict[Move, int]:
        return {m: (root.children[m].N if m in root.children else 0) for m in root.moves}

    def select_move(self, root: Node, temperature: float = 1.0) -> Move:
        counts = self.visit_counts(root)
        moves = list(counts)
        if temperature <= 1e-6:
            return max(moves, key=lambda m: (counts[m], -root.children[m].Q if m in root.children else 0))
        weights = [counts[m] ** (1.0 / temperature) for m in moves]
        if sum(weights) == 0:
            return self.rng.choice(moves)
        return self.rng.choices(moves, weights=weights)[0]

    def policy_target(self, root: Node) -> dict[Move, float]:
        counts = self.visit_counts(root)
        tot = sum(counts.values()) or 1
        return {m: c / tot for m, c in counts.items()}

    @staticmethod
    def advance(root: Node, mv: Move) -> Node:
        """Reuse the subtree under ``mv`` as the next root."""
        child = root.children.get(mv)
        if child is None:
            return Node(apply_move(root.pos, mv))
        child.vloss = 0
        return child


def mcts_policy(mcts: MCTS, temperature: float = 0.0):
    """Adapter to :mod:`dq.game`'s Policy signature."""
    def policy(pos: Position, moves: list[Move]) -> Move:
        root = mcts.search(pos, add_noise=temperature > 0)
        return mcts.select_move(root, temperature)
    return policy
