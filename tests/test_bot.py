import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bot"))

from dq_engine import DQSearcher, to_board_move  # noqa: E402

from dq.model import ModelConfig
from dq.ref_pydraughts import RefBoard
from dq.rules import Position, legal_moves
from dq.tablebase import Tablebase

DATA_TB = Path(__file__).resolve().parents[1] / "data" / "tb"


def test_searcher_returns_legal_move_and_pydraughts_accepts_it() -> None:
    s = DQSearcher(weights="", sims=16, tb_dir="", device=torch.device("cpu"),
                   mcfg=ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))
    pos = Position.start()
    ref = RefBoard(pos)
    for _ in range(6):
        mv = s.choose(pos)
        assert mv in legal_moves(pos)
        ref.push(mv)                         # pydraughts accepts the [from,to] steps
        from dq.rules import apply_move
        pos = apply_move(pos, mv)
        assert ref.position() == pos
    assert to_board_move(mv) == [[a, b] for a, b in zip(mv.path, mv.path[1:])]


@pytest.mark.skipif(not DATA_TB.exists(), reason="no tablebase on disk")
def test_tablebase_move_wins_fastest() -> None:
    s = DQSearcher(weights="", sims=4, tb_dir=str(DATA_TB), device=torch.device("cpu"),
                   mcfg=ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))
    pos = Position.from_fen("W:WK18:BK4")     # win in 3 via 18-29 (see test_tablebase)
    mv = s.choose(pos)
    assert str(mv) == "18-29"
