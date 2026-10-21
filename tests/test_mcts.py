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


def test_batched_search_matches_single_on_tablebase_wins(tb3: Tablebase) -> None:
    """search_batch over many trees with virtual loss must still keep wins."""
    rng = random.Random(21)
    m = MCTS(uniform_evaluator, SearchConfig(simulations=60, dirichlet_eps=0.0, leaves_per_tree=8),
             tablebase=tb3, rng=rng)
    from dq.mcts import Node
    roots = []
    while len(roots) < 24:
        p = random_position(rng, rng.randint(1, 2), rng.randint(1, 2), king_prob=0.5)
        if tb3.covers(p) and len(legal_moves(p)) >= 2 and tb3.probe(p)[0] == WIN:
            roots.append(Node(p))
    m.search_batch(roots, add_noise=False)
    for r in roots:
        assert r.N == 60 and r.vloss == 0
        mv = m.select_move(r, 0.0)
        child = apply_move(r.pos, mv)
        cv = (LOSS, 0) if child.pieces(child.turn) == 0 else tb3.probe(child)
        assert cv[0] == LOSS, (r.pos.fen(), str(mv), cv)


def test_tree_reuse_advance() -> None:
    m = MCTS(uniform_evaluator, SearchConfig(simulations=30, dirichlet_eps=0.0), rng=random.Random(2))
    root = m.search(Position.start(), add_noise=False)
    mv = m.select_move(root, 0.0)
    child = MCTS.advance(root, mv)
    assert child.pos == apply_move(root.pos, mv) and child.N > 0
    m.search(child.pos, root=child, add_noise=False)
    assert child.N >= 30


def test_virtual_loss_diversifies_leaves() -> None:
    """In one batched round the 8 descents from a fresh root must reach
    8 distinct leaves (virtual loss repels in-flight paths)."""
    from dq.mcts import Node
    seen = []
    def ev(positions, moves):
        seen.extend(positions)
        return uniform_evaluator(positions, moves)
    m = MCTS(ev, SearchConfig(simulations=8, dirichlet_eps=0.0, leaves_per_tree=8), rng=random.Random(0))
    m.search_batch([Node(Position.start())], add_noise=False)
    # root first, then one round: the start position has 7 legal moves, so
    # 8 descents reach all 7 children (the 8th repeats one and is deduped)
    assert len(seen) == 1 + 7 and len(set(p.fen() for p in seen[1:])) == 7


def test_repetition_in_tree_is_scored_as_draw() -> None:
    """A child position already seen twice in the game (so a move there would be the
    third occurrence) is backed up as a draw and never expanded; others are expanded."""
    from dq.rules import zobrist
    pos = Position.start()
    moves = legal_moves(pos)
    target = moves[0]
    child_pos = apply_move(pos, target)
    seen = {zobrist(pos): 1, zobrist(child_pos): 2}
    m = MCTS(uniform_evaluator, SearchConfig(simulations=80, dirichlet_eps=0.0), rng=random.Random(0))
    root = m.search(pos, add_noise=False, seen=seen)
    rep = root.children[target]
    assert rep.N > 0 and rep.prior == {} and rep.W == 0.0      # visited, drawn, not expanded
    assert any(c.prior for mv, c in root.children.items() if mv != target)
    # without `seen` the same child is expanded normally
    root2 = m.search(pos, add_noise=False)
    assert root2.children[target].prior


def test_tablebase_leaf_respects_draw_ply_limit() -> None:
    """A tablebase win whose DTM exceeds the active draw rule's ply limit is scored 0, not
    +1 (e.g. 7.2.8: 10 plies for <=2 pieces incl. a king vs a lone king).  Real 3-piece
    tables have no such position (2v1 wins are short), so a stub tablebase is used."""
    from dq.game import draw_ply_limit
    from dq.mcts import Node
    from dq.rules import Position

    class StubTB:
        def covers(self, pos): return True
        def probe(self, pos): return (WIN, 50)

    from dq.rules import bit
    p = Position(0, bit(18) | bit(22), 0, bit(1))      # two white kings vs a lone black king
    assert draw_ply_limit(p) == 10
    assert MCTS(uniform_evaluator, tablebase=StubTB())._prepare(Node(p)) == 1.0
    assert MCTS(uniform_evaluator, tablebase=StubTB(), tb_ply_limit=draw_ply_limit)._prepare(Node(p)) == 0.0
    # a position with no counting rule is untouched
    assert MCTS(uniform_evaluator, tablebase=StubTB(), tb_ply_limit=draw_ply_limit)._prepare(Node(Position.start())) == 1.0
