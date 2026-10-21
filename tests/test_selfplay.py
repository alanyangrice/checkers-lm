import random

import pytest

torch = pytest.importorskip("torch")

from dq.model import ModelConfig, NetEvaluator, SquareTransformer
from dq.selfplay import match, tablebase_wdl_error
from dq.tablebase import Tablebase


def _net(seed: int) -> SquareTransformer:
    torch.manual_seed(seed)
    return SquareTransformer(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128)).eval()


def test_batched_match_is_symmetric_and_handles_last_capture() -> None:
    """Games that end mid-round (a capture of the last piece) must not be
    searched again; A-vs-B and B-vs-A must mirror (search is deterministic
    at T=0 without noise)."""
    tb = Tablebase(); tb.build(3)
    a, b = NetEvaluator(_net(1)), NetEvaluator(_net(2))
    ab = match(a, b, games=20, sims=16, tb=tb, rng=random.Random(0), max_plies=300)
    ba = match(b, a, games=20, sims=16, tb=tb, rng=random.Random(0), max_plies=300)
    assert abs(ab + ba - 1.0) < 1e-9
    aa = match(a, a, games=10, sims=8, tb=None, rng=random.Random(0), max_plies=300)
    assert aa == 0.5   # paired openings + deterministic search => exact mirror
    s_raw = match(a, b, games=4, sims=8, tb=None, rng=random.Random(0), max_plies=120, raw_a=True)
    assert 0.0 <= s_raw <= 1.0


def test_tablebase_wdl_error_in_range() -> None:
    tb = Tablebase(); tb.build(2)
    e = tablebase_wdl_error(_net(3), tb, 100, random.Random(1), torch.device("cpu"))
    assert 0.0 <= e <= 1.0
