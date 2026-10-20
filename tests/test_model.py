import random

import pytest

torch = pytest.importorskip("torch")

from dq.mcts import MCTS, SearchConfig
from dq.model import (
    MAX_CAP, N_PAIRS, Example, ModelConfig, NetEvaluator, SquareTransformer, collate, loss_fn,
    make_example, pair_index,
)
from dq.rules import Position, legal_moves, random_position
from dq.tokens import pov_move


def small() -> SquareTransformer:
    torch.manual_seed(0)
    return SquareTransformer(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))


def test_forward_shapes() -> None:
    m = small()
    ids = torch.randint(0, 5, (3, 32))
    out = m(ids)
    assert out["policy"].shape == (3, N_PAIRS)
    assert out["wdl"].shape == (3, 3)
    assert out["ncap"].shape == (3, MAX_CAP + 1)
    assert len(m.hidden_states(ids)) == 3


def test_pair_index_unique_for_quiet_moves() -> None:
    rng = random.Random(0)
    for _ in range(50):
        p = random_position(rng, rng.randint(1, 6), rng.randint(1, 6), king_prob=0.3)
        mv = [pov_move(m, p.turn) for m in legal_moves(p)]
        if mv and not mv[0].is_capture:
            idx = [pair_index(m) for m in mv]
            assert len(set(idx)) == len(idx)
            assert all(0 <= k < N_PAIRS for k in idx)


def test_make_example_and_collate() -> None:
    p = Position.start()
    moves = legal_moves(p)
    pol = {m: 1.0 / len(moves) for m in moves}
    e = make_example(p, pol, 1.0)
    assert len(e.ids) == 32 and abs(sum(e.policy.values()) - 1.0) < 1e-6 and e.ncap == 0
    b = collate([e, e])
    assert b["policy"].shape == (2, N_PAIRS) and b["wdl"].tolist() == [0, 0]
    m = small()
    loss, parts = loss_fn(m(b["ids"]), b)
    assert loss.item() > 0 and set(parts) == {"policy", "wdl", "ncap"}


def test_evaluator_priors_sum_to_one_and_mcts_runs() -> None:
    m = small()
    ev = NetEvaluator(m)
    rng = random.Random(2)
    for _ in range(10):
        p = random_position(rng, rng.randint(1, 5), rng.randint(1, 5), king_prob=0.4)
        moves = legal_moves(p)
        if not moves:
            continue
        (prior, v), = ev([p], [moves])
        assert abs(sum(prior.values()) - 1.0) < 1e-5 and -1.0 <= v <= 1.0
    s = MCTS(ev, SearchConfig(simulations=20, dirichlet_eps=0.0), rng=rng)
    root = s.search(Position.start(), add_noise=False)
    assert sum(s.visit_counts(root).values()) == 20


def test_overfits_tiny_batch() -> None:
    """Sanity: the model can memorise a handful of examples."""
    rng = random.Random(4)
    exs = []
    for _ in range(8):
        p = random_position(rng, 4, 4, king_prob=0.2)
        moves = legal_moves(p)
        if not moves:
            continue
        exs.append(make_example(p, {moves[0]: 1.0}, rng.choice([-1.0, 0.0, 1.0])))
    m = small().train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    b = collate(exs)
    first = None
    for _ in range(150):
        loss, _ = loss_fn(m(b["ids"]), b)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < first * 0.2
