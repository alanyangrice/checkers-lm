"""PUCT Monte-Carlo tree search over :mod:`dq.rules` (AlphaZero-style).

The search is independent of the network.  An ``Evaluator`` maps a
position to ``(priors, value)`` where ``priors`` is a dict Move -> prob over
the legal moves and ``value`` is in [-1, 1] from the point of view of the
side to move.  Batched evaluation is the evaluator's business; the tree
asks for one position at a time (virtual loss / batching can be added
later without changing the interface).

Terminal handling: a position with no legal move is a loss (-1) for the
side to move.  If a :class:`dq.tablebase.Tablebase` is supplied, covered
positions are scored exactly (+1 / 0 / -1) and not expanded.

``search`` returns the root's visit distribution, which is the policy
target for self-play; ``select_move`` samples from it with a temperature.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from dq.rules import Move, Position, apply_move, legal_moves

Evaluator = Callable[[Position, list[Move]], tuple[dict[Move, float], float]]


def uniform_evaluator(pos: Position, moves: list[Move]) -> tuple[dict[Move, float], float]:
    p = 1.0 / len(moves)
    return {m: p for m in moves}, 0.0


@dataclass
class Node:
    pos: Position
    moves: list[Move] = field(default_factory=list)
    prior: dict[Move, float] = field(default_factory=dict)
    children: dict[Move, "Node"] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0          # sum of values from the perspective of the side to move at this node
    terminal_value: Optional[float] = None   # set for terminal / tablebase nodes

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0

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


class MCTS:
    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None,
                 tablebase=None, rng: random.Random | None = None) -> None:
        self.eval = evaluator
        self.cfg = config or SearchConfig()
        self.tb = tablebase
        self.rng = rng or random.Random()

    # -- leaf evaluation --------------------------------------------------
    def _evaluate(self, node: Node, use_tb: bool = True) -> float:
        """Expand ``node`` (if needed) and return its value for the side to move.
        ``use_tb=False`` forces expansion even if the tablebase covers the
        position (needed at the root, where a move must be chosen)."""
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
        prior, value = self.eval(node.pos, node.moves)
        node.prior = prior
        return value

    def _add_root_noise(self, root: Node) -> None:
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        if eps <= 0 or len(root.moves) <= 1:
            return
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in root.moves]
        s = sum(noise)
        for m, n in zip(root.moves, noise):
            root.prior[m] = (1 - eps) * root.prior[m] + eps * n / s

    # -- selection ----------------------------------------------------------
    def _select(self, node: Node) -> Move:
        sqrt_n = math.sqrt(node.N + 1)
        best, best_score = None, -1e9
        for m in node.moves:
            child = node.children.get(m)
            # child's Q is from the child's mover's perspective -> negate
            q = -child.Q if (child is not None and child.N) else self.cfg.fpu
            u = self.cfg.c_puct * node.prior[m] * sqrt_n / (1 + (child.N if child else 0))
            score = q + u
            if score > best_score:
                best, best_score = m, score
        return best  # type: ignore[return-value]

    # -- search ---------------------------------------------------------------
    def search(self, pos: Position, root: Node | None = None, add_noise: bool = True) -> Node:
        root = root or Node(pos)
        if not root.expanded or root.terminal_value is not None:
            root.terminal_value = None
            self._evaluate(root, use_tb=False)
        if add_noise:
            self._add_root_noise(root)
        for _ in range(self.cfg.simulations):
            node = root
            path = [root]
            # descend
            while node.expanded and node.terminal_value is None:
                m = self._select(node)
                child = node.children.get(m)
                if child is None:
                    child = Node(apply_move(node.pos, m))
                    node.children[m] = child
                node = child
                path.append(node)
            value = self._evaluate(node)   # from node's mover's perspective
            # backup: alternate sign up the path
            for n in reversed(path):
                n.N += 1
                n.W += value
                value = -value
        return root

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


def mcts_policy(mcts: MCTS, temperature: float = 0.0):
    """Adapter to :mod:`dq.game`'s Policy signature."""
    def policy(pos: Position, moves: list[Move]) -> Move:
        root = mcts.search(pos, add_noise=temperature > 0)
        return mcts.select_move(root, temperature)
    return policy
