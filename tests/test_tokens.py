import random

import pytest

from dq.rules import BLACK, WHITE, Position, apply_move, legal_moves, random_position
from dq.tokens import (
    BOARD_VOCAB, EMPTY_TOK, MK, MM, OK, OM, SPECIAL_TOKENS, board_to_text, decode_board,
    encode_board, legal_moves_pov, move_to_text, pov_move, text_to_move,
)


def test_start_position_tokens() -> None:
    ids = encode_board(Position.start())
    assert len(ids) == 32
    assert ids[:12] == [OM] * 12
    assert ids[12:20] == [EMPTY_TOK] * 8
    assert ids[20:] == [MM] * 12


def test_black_to_move_is_rotated() -> None:
    pos = apply_move(Position.start(), legal_moves(Position.start())[0])
    assert pos.turn == BLACK
    ids = encode_board(pos)
    # after rotation, the mover (Black) is "me", at the bottom, moving toward square 1
    assert ids[20:].count(MM) == 12
    assert ids[:12].count(OM) == 11 and ids[:20].count(EMPTY_TOK) + ids[:20].count(OM) == 20


@pytest.mark.parametrize("seed", range(50))
def test_roundtrip(seed: int) -> None:
    rng = random.Random(seed)
    pos = random_position(rng, rng.randint(0, 10), rng.randint(0, 10), king_prob=0.4)
    assert decode_board(encode_board(pos), pos.turn) == pos


def test_vocab_sizes() -> None:
    assert BOARD_VOCAB == 5
    assert len(SPECIAL_TOKENS) == 32 + 4 + 2
    assert len(set(SPECIAL_TOKENS)) == len(SPECIAL_TOKENS)


@pytest.mark.parametrize("seed", range(30))
def test_move_text_roundtrip(seed: int) -> None:
    rng = random.Random(100 + seed)
    pos = random_position(rng, rng.randint(1, 8), rng.randint(1, 8), king_prob=0.5)
    legal = legal_moves_pov(pos)
    for mv in legal:
        assert text_to_move(move_to_text(mv), legal) == mv


def test_pov_move_involution() -> None:
    rng = random.Random(7)
    pos = random_position(rng, 5, 5, turn=BLACK)
    for mv in legal_moves(pos):
        assert pov_move(pov_move(mv, BLACK), BLACK) == mv
        assert pov_move(mv, WHITE) == mv


def test_pov_moves_are_legal_in_pov_position() -> None:
    rng = random.Random(9)
    for _ in range(40):
        pos = random_position(rng, rng.randint(1, 8), rng.randint(1, 8), turn=BLACK, king_prob=0.4)
        assert sorted(legal_moves_pov(pos)) == sorted(legal_moves(pos.pov()))


def test_board_to_text() -> None:
    t = board_to_text(Position.from_fen("W:W21,K22:B1"))
    assert t == "<PIECE_OM><SQ_1> <PIECE_MM><SQ_21> <PIECE_MK><SQ_22>"
    t = board_to_text(Position.from_fen("B:W21:B1"))   # rotated: black man at 1 -> my man at 32
    assert t == "<PIECE_OM><SQ_12> <PIECE_MM><SQ_32>"
