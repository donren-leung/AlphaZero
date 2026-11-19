from typing import Sequence
import io
import random

import chess
import chess.pgn

import numpy as np
from numpy.typing import NDArray
import pytest

from alphazero.games.ChessUtils import (
    AUXILIARY_PLANE_START,
    CHESS_ACTION_SIZE,
    CHESS_ENCODING_VERSION,
    CHESS_PIECE_LAYERS,
    CHESS_INPUT_CHANNELS,
    REPETITION_PLANE_START,
    TIME_STEP_LAYERS,
    TIME_STEPS,
    encode_move,
    decode_action,
    encode_chess,
    initial_history
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
    with pytest.raises(ValueError, match="off the board"):
        # position 0, plane 7
        # source_file, source_rank = 0, 0 = a1
        # QUEEN_DIRECTIONS[7] is (1, -1) or north-west
        decode_action(board, 7, 1)
    with pytest.raises(ValueError, match="off the board"):
        # h8 northward (to h9) is off-board.
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
def test_full_action_space(fen: str) -> None:
    """
    Tests:
    1. Every legal move encodes to a unique action.
    2. Every legal action decodes to its original move.
    3. Every action in the full 4,672-action space either decodes to an on-board move
       or raises ValueError because it points off the board.
    """
    board = chess.Board(fen)
    player = _player(board)

    expected = {
        encode_move(board, move, player): move
        for move in board.legal_moves
    }
    assert len(expected) == board.legal_moves.count()

    for action in range(CHESS_ACTION_SIZE):
        try:
            move = decode_action(board, action, player)
        except ValueError as exc:
            assert "off the board" in str(exc)
            assert action not in expected
        else:
            # Every successfully decoded action must remain on the board.
            assert 0 <= move.from_square < 64
            assert 0 <= move.to_square < 64

            # Every legal action must round-trip correctly.
            if action in expected:
                assert move == expected[action]

def old_encode_chess(board: chess.Board, repetitions: Sequence[int] | None=None) -> NDArray[np.uint8]:
    """
    Accepts optional repetition history.
    repetitions[t] is the number of repetitions of pos t halfmoves ago.
    """
    # AlphaZero encoding
    # P1 pieces (6)
    # P2 pieces (6)
    # Repetition 2, 3 (2); whole plane binary mask
    # Repeat for T=8 halfmove history (8 * 14 = 112)

    # P1 castling rights (2); whole plane binary mask
    # P2 castling rights (2); whole plane binary mask
    # Total move count (1); whole plane encoded
    # No progress clock (1); whole plane encoded

    is_white = board.turn

    # Time step planes
    features = np.zeros((CHESS_INPUT_CHANNELS, 8, 8), dtype=np.uint8)
    history = board.copy()
    for ply in range(TIME_STEPS):
        for square, piece in history.piece_map().items():
            oriented = square if is_white else chess.square_mirror(square)
            side = 0 if piece.color == is_white else CHESS_PIECE_LAYERS
            plane = ply * TIME_STEP_LAYERS + side + piece.piece_type - 1
            features[plane, chess.square_rank(oriented),
                     chess.square_file(oriented)] = 1

        if repetitions is not None:
            assert ply < len(repetitions), "Repetition history is shorter than the number of halfmoves"
            r2 = repetitions[ply] >= 2
            r3 = repetitions[ply] >= 3
        else:
            r2 = history.is_repetition(2)
            r3 = history.is_repetition(3)

        features[ply * TIME_STEP_LAYERS + REPETITION_PLANE_START, :, :] = r2
        features[ply * TIME_STEP_LAYERS + REPETITION_PLANE_START + 1, :, :] = r3

        if not history.move_stack:
            break
        history.pop()

    # Auxiliary planes
    for plane, color, kingside in (
        (AUXILIARY_PLANE_START, is_white, True), (AUXILIARY_PLANE_START + 1, is_white, False),
        (AUXILIARY_PLANE_START + 2, not is_white, True), (AUXILIARY_PLANE_START + 3, not is_white, False),
    ):
        has_right = (
            board.has_kingside_castling_rights(color)
            if kingside else board.has_queenside_castling_rights(color)
        )
        features[plane, :, :] = has_right

    features[AUXILIARY_PLANE_START + 4, 0, 0] = board.ply() >> 8
    features[AUXILIARY_PLANE_START + 4, 0, 1] = board.ply() & 0xFF
    features[AUXILIARY_PLANE_START + 5, :, :] = board.halfmove_clock

    return features

@pytest.mark.parametrize(
    ("pgn_str"),
    [
        pytest.param(
            """
            1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# *
            """,
            id="immortal-game",
        ),
        pytest.param(
            """
            1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5 6.Bc4 Nf6 7.Qb3 Qe7 8.Nc3
            c6 9.Bg5 b5 10.Nxb5 cxb5 11.Bxb5+ Nbd7 12.O-O-O Rd8 13.Rxd7 Rxd7 14.Rd1 Qe6 15.Bxd7+ Nxd7
            16.Qb8+ Nxb8 17.Rd8# 1-0
            """,
            id="opera-game",
        ),
        pytest.param(
            """
            1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. d4 Nd6 6. dxe5 Nxb5 7. a4 Nbd4 8. Nxd4 Nxd4 9. Qxd4 d5 10. exd6 Qxd6 11. Qe4+ Qe6 12. Qd4 Qd6 13. Qe4+ Qe6 14. Qd4 Qd6 15. Qe4+ Qe6 16. Qd4 *
            """,
            id="berlin-draw",
        ),
    ]
)
def test_new_encode_chess(pgn_str: str) -> None:
    pgn = io.StringIO(pgn_str.strip())
    game = chess.pgn.read_game(pgn)
    assert game is not None

    board = game.board()

    enc1 = encode_chess(board, initial_history(board))
    enc2 = old_encode_chess(board)
    assert np.array_equal(enc1, enc2)

    for move in game.mainline_moves():
        board.push(move)
        enc1 = encode_chess(board, initial_history(board))
        enc2 = old_encode_chess(board)
        assert np.array_equal(enc1, enc2)

    assert board.outcome(claim_draw=True) is not None
