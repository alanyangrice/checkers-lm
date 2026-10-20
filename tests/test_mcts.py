import random

import pytest

from dq.game import play_game, random_policy
from dq.mcts import MCTS, SearchConfig, mcts_policy, uniform_evaluator
from dq.rules import Position, apply_move, legal_moves, random_position
from dq.tablebase import LOSS, WIN, Tablebase


@pytest.fixture(scope="module")
def tb3() -> Tablebase:
    tb = Tablebase()
    tb.build(3)
    return tb


def test_visit_counts_sum_to_simulations() -> None:
    m = MCTS(uniform_evaluator, SearchConfig(simulations=50, dirichlet_eps=0.0), rng=random.Random(0))
    root = m.search(Position.start(), add_noise=False)
    assert sum(m.visit_counts(root).values()) == 50
    assert root.N == 50


def test_finds_immediate_win() -> None:
    # White king can capture Black's last piece: every search must pick it.
    pos = Position.from_fen("W:WK29,K32:B25")
    m = MCTS(uniform_evaluator, SearchConfig(simulations=30, dirichlet_eps=0.0), rng=random.Random(1))
    mv = m.select_move(m.search(pos, add_noise=False), temperature=0.0)
    assert mv.is_capture


def test_mcts_with_tablebase_plays_tablebase_wins(tb3: Tablebase) -> None:
    """From a 3-piece WIN position, MCTS leaves must be scored by the
    tablebase, so the chosen move must keep the win (child is LOSS for the
    opponent)."""
    rng = random.Random(5)
    m = MCTS(uniform_evaluator, SearchConfig(simulations=60, dirichlet_eps=0.0), tablebase=tb3, rng=rng)
    tried = 0
    for _ in range(400):
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if not tb3.covers(p) or len(legal_moves(p)) < 2:
            continue
        v, d = tb3.probe(p)
        if v != WIN:
            continue
        root = m.search(p, add_noise=False)
        mv = m.select_move(root, 0.0)
        child = apply_move(p, mv)
        cv = (LOSS, 0) if child.pieces(child.turn) == 0 else tb3.probe(child)
        assert cv[0] == LOSS, (p.fen(), str(mv), cv)
        tried += 1
        if tried >= 40:
            break
    assert tried >= 40


def test_mcts_beats_random_in_full_games() -> None:
    rng = random.Random(9)
    m = MCTS(uniform_evaluator, SearchConfig(simulations=40, dirichlet_eps=0.0), rng=rng)
    pol = mcts_policy(m, temperature=0.0)
    score = 0.0
    n = 6
    for i in range(n):
        if i % 2 == 0:
            g = play_game(pol, random_policy(rng), max_plies=300)
            score += g.result
        else:
            g = play_game(random_policy(rng), pol, max_plies=300)
            score += 1.0 - g.result
    assert score / n > 0.6, score
