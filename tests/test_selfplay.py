import random

import pytest

torch = pytest.importorskip("torch")

from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.selfplay import match, tablebase_wdl_error
from dq.tablebase import Tablebase


def _net(seed: int) -> SquareTransformer:
    torch.manual_seed(seed)
    return SquareTransformer(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)).eval()


def test_batched_match_runs_and_scores_in_range() -> None:
    rng = random.Random(0)
    a, b = NetEvaluator(_net(1)), NetEvaluator(_net(2))
    s = match(a, b, games=6, sims=8, tb=None, rng=rng, max_plies=120)
    assert 0.0 <= s <= 1.0
    s_raw = match(a, b, games=4, sims=8, tb=None, rng=rng, max_plies=120, raw_a=True)
    assert 0.0 <= s_raw <= 1.0


def test_tablebase_wdl_error_in_range() -> None:
    tb = Tablebase(); tb.build(2)
    e = tablebase_wdl_error(_net(3), tb, 100, random.Random(1), torch.device("cpu"))
    assert 0.0 <= e <= 1.0
