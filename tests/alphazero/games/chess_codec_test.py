import random

import chess
import pytest
from alphazero.games.ChessUtils import (
    CHESS_ACTION_SIZE, CHESS_ENCODING_VERSION, encode_move, decode_action,
)

def _player(board: chess.Board) -> int:
    return 1 if board.turn == chess.WHITE else -1


def _encode_uci(board: chess.Board, uci: str) -> int:
    return encode_move(board, chess.Move.from_uci(uci), _player(board))

def _roundtrip(board):
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
        _roundtrip(board)
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
    _roundtrip(chess.Board(fen))

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

@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
    ],
)
def test_colour_mirror_invariance(fen: str) -> None:
    """Mirroring colours/ranks preserves the policy action integer."""
    white_board = chess.Board(fen)
    assert white_board.turn == chess.WHITE
    black_board = white_board.mirror()
    assert black_board.turn == chess.BLACK

    for white_move in white_board.legal_moves:
        black_move = chess.Move(
            chess.square_mirror(white_move.from_square),
            chess.square_mirror(white_move.to_square),
            promotion=white_move.promotion,
        )
        assert black_move in black_board.legal_moves
        action = encode_move(white_board, white_move, 1)
        assert encode_move(black_board, black_move, -1) == action
        assert decode_action(black_board, action, -1) == black_move


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    ],
)
def test_full_action_space_decodes_only_legal_moves(fen: str) -> None:
    """Test all 4,672 possible actions, not only actions emitted by encode_move."""
    board = chess.Board(fen)
    player = _player(board)
    expected = {encode_move(board, move, player): move for move in board.legal_moves}
    assert len(expected) == board.legal_moves.count()

    for action in range(CHESS_ACTION_SIZE):
        if action in expected:
            assert decode_action(board, action, player) == expected[action]
        else:
            try:
                move = decode_action(board, action, player)
            except ValueError:
                pass
            else:
                pytest.fail(
                    f"Action {action} unexpectedly decoded to "
                    f"{move.uci()} ({board.san(move)}), "
                    f"but was not produced by encode_move()"
                )
