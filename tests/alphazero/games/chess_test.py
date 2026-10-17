
    
  
"""Run with: pytest -q test_chess_action_codec.py"""

import chess
import pytest
from alphazero.games.ChessUtils import (
    CHESS_ACTION_SIZE, CHESS_ENCODING_VERSION, encode_move, decode_action,
)


def roundtrip(board):
    player = 1 if board.turn == chess.WHITE else -1
    actions = [encode_move(board, move, player) for move in board.legal_moves]
    assert len(actions) == len(set(actions)), "legal actions collided"
    assert all(0 <= a < CHESS_ACTION_SIZE for a in actions)
    for move, action in zip(list(board.legal_moves), actions):
        assert decode_action(board, action, player) == move


def test_constants():
    assert CHESS_ACTION_SIZE == 4672
    assert CHESS_ENCODING_VERSION == 1


def test_initial_and_random_play():
    board = chess.Board()
    for _ in range(180):
        roundtrip(board)
        if board.is_game_over():
            board.reset()
        else:
            board.push(next(iter(board.legal_moves)))


@pytest.mark.parametrize("fen", [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", # starting position
    "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", # Berlin
    "r1bqk2r/ppp1bppp/2np1n2/1B2p3/4P3/2P2N2/PP1P1PPP/RNBQR1K1 b kq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # white castling
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",  # black castling
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",  # white en passant
    "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",  # black en passant
    "4k3/P6P/8/8/8/8/8/4K3 w - - 0 1",  # white promotions
    "4k3/8/8/8/8/8/p6p/4K3 b - - 0 1",  # black promotions
    "1r2k2r/P6P/8/8/8/8/8/4K3 w k - 0 1",  # promotion captures
])
def test_special_positions(fen):
    roundtrip(chess.Board(fen))


def test_invalid():
    board = chess.Board()
    with pytest.raises(AssertionError):
        encode_move(board, chess.Move.from_uci("e2e4"), -1)
    with pytest.raises(AssertionError):
        decode_action(board, 0, 0)
    for action in (-1, 4672, 999999,-58):
        with pytest.raises(ValueError):
            decode_action(board, action, 1)
    with pytest.raises(ValueError):
        encode_move(board, chess.Move.from_uci("e2e5"), 1)
    with pytest.raises(ValueError):
        decode_action(board, 7, 1)  # a1 -> b2: no legal move
    with pytest.raises(ValueError):
        decode_action(board, 0, 1)  # a1 -> a2: illegal occupied square
    # h8 northward is off-board.
    with pytest.raises(ValueError, match="off the board"):
        decode_action(board, (7 * 8 + 7) * 73, 1)
