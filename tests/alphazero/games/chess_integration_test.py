"""
Tests for the ChessState/ChessGame wrappers.
"""
import io

import chess
import chess.pgn
import numpy as np
import pytest

from alphazero.games.Chess import ChessGame, ChessState
from alphazero.games.ChessUtils import CHESS_ACTION_SIZE, encode_move

def _player(board: chess.Board) -> int:
    return 1 if board.turn == chess.WHITE else -1

@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        chess.STARTING_FEN.replace(" w ", " b "),
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    ],
)
def test_legal_mask_matches_codec_exactly(fen: str) -> None:
    board = chess.Board(fen)
    state = ChessState(board)
    mask = state.get_legal_actions()

    expected_actions = {
        encode_move(board, move, _player(board)) for move in board.legal_moves
    }
    assert mask.shape == (CHESS_ACTION_SIZE,)
    assert mask.dtype == np.dtype(np.bool_)
    assert {int(i) for i in np.flatnonzero(mask)} == expected_actions
    assert np.count_nonzero(mask) == board.legal_moves.count()

def test_next_state_copy_does_not_mutate_original_board() -> None:
    state = ChessState(chess.Board())
    original_fen = state.board.fen()
    original_board = state.board
    action = encode_move(original_board, chess.Move.from_uci("e2e4"), 1)

    next_state = state.get_next_state(action, player=1, copy=True)

    assert state.board is original_board
    assert state.board.fen() == original_fen
    assert next_state is not state
    assert next_state.board is not original_board
    assert next_state.board.peek().uci() == "e2e4"
    assert next_state.board.turn == chess.BLACK

def test_next_state_without_copy_mutates_original_board() -> None:
    state = ChessState(chess.Board())
    original_board = state.board
    action = encode_move(original_board, chess.Move.from_uci("e2e4"), 1)

    next_state = state.get_next_state(action, player=1, copy=False)

    assert next_state.board is original_board
    assert original_board.peek().uci() == "e2e4"
    assert original_board.turn == chess.BLACK

def test_next_state_rejects_wrong_player() -> None:
    state = ChessState(chess.Board())
    initial_fen = state.board.fen()
    legal_action = encode_move(state.board, chess.Move.from_uci("e2e4"), 1)

    with pytest.raises(AssertionError):
        state.get_next_state(legal_action, player=-1)
    assert state.board.fen() == initial_fen

@pytest.mark.parametrize(
    ("fen", "move_uci", "expected"),
    [
        pytest.param(
            chess.STARTING_FEN,
            "e2e4",
            (0, False),
            id="opening-e4-not-terminal",
        ),
        pytest.param(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "h5f7",
            (1, True),
            id="scholars-mate-qxf7-checkmate",
        ),
        pytest.param(
            "7k/5K2/8/6Q1/8/8/8/8 w - - 0 1",
            "g5g6",
            (0, True),
            id="qg6-stalemate",
        ),
        pytest.param(
            "4k3/8/8/8/8/8/4K3/4q3 w - - 0 1",
            "e2e1",
            (0, True),
            id="kxe1-insufficient-material",
        ),
        pytest.param(
            "r4b1r/ppp3pp/8/4p3/2Pq4/3P4/PP2QPPP/2k1K2R w K - 0 18",
            "e1g1",
            (1, True),
            id="morphys-castling-checkmate",
        ),
    ],
)
def test_state_value_and_termination_after_move(
    fen: str, move_uci: str, expected: tuple[int, bool]
) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)

    assert not board.is_game_over(claim_draw=True)
    assert move in board.legal_moves

    player = _player(board)
    action = encode_move(board, move, player)
    next_state = ChessState(board).get_next_state(action, player)

    assert next_state.board.peek() == move
    assert next_state.get_value_and_terminated() == expected

def test_threefold_repetition_is_claimable_draw() -> None:
    board = chess.Board()
    assert not board.can_claim_threefold_repetition()
    assert ChessState(board).get_value_and_terminated() == (0, False)

    for _ in range(2):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)

    assert board.can_claim_threefold_repetition()
    assert ChessState(board).get_value_and_terminated() == (0, True)

def test_fifty_move_rule_is_claimable_draw() -> None:
    # White cannot claim 50-move rule yet, since even if he plays a move, black still has the right to play his 50th move.
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 98 1")
    assert not board.can_claim_fifty_moves()
    assert ChessState(board).get_value_and_terminated() == (0, False)

    # In the same position, black can claim 50-move rule by announcing a non-zeroing move that would tick over to 100 half-moves.
    board = chess.Board("4k3/8/8/8/8/8/8/R3K3 b Q - 99 1")
    assert board.can_claim_fifty_moves()
    assert ChessState(board).get_value_and_terminated() == (0, True)

    # An exceptional position to the above case: white's king is trapped in the corner and must push his pawn, resetting the clock.
    board = chess.Board("8/8/6k1/4p3/8/1q2P3/8/K7 w - - 99 1")
    assert not board.can_claim_fifty_moves()
    assert ChessState(board).get_value_and_terminated() == (0, False)

def test_game_move_updates_state_and_changes_starting_board() -> None:
    start = chess.Board()
    game = ChessGame(ChessState(start))
    move = chess.Move.from_uci("e2e4")
    action = encode_move(start, move, 1)
    initial_fen = start.fen()

    assert game.get_legal_actions()[action]
    game._make_move(action, 1)

    assert start.fen() != initial_fen
    assert game.state.board.peek() == move
    assert game.state.board.turn == chess.BLACK
    assert game.get_legal_actions().shape == (CHESS_ACTION_SIZE,)

def test_compact_repr_roundtrip() -> None:
    game = ChessGame()
    for i in range(1, 3):
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            move = game.parse_move_str(san)
            game.make_move(move)

            if i == 1 or san in ("Nf3", "Nf6"):
                assert game.state.get_value_and_terminated() == (0, False)

            compact = game.state.to_compact_repr(game.current_player)
            restored = compact.unpack()

            expected = game.state.encode_for_model()
            wrong_channels = np.flatnonzero(
                np.any(restored != expected, axis=(1, 2))
            )
            print("Mismatched channels:", wrong_channels)

            np.testing.assert_array_equal(
                restored,
                game.state.encode_for_model(),
                err_msg=f"Compact representation roundtrip failed after move {san}, {i}th rep\n{str(game.state.board)}",
            )

    encoded = game.state.encode_for_model()

    assert np.all(encoded[12] == 1)           # Current: second occurrence or later.
    assert np.all(encoded[13] == 1)           # Current: third occurrence.
    assert np.all(encoded[4 * 14 + 12] == 1)  # Four plies ago: second occurrence.
    assert np.all(encoded[4 * 14 + 13] == 0)  # Four plies ago: not third.

@pytest.mark.parametrize(
    ("pgn_str", "expected_outcome"),
    [
        pytest.param(
            """
            1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# *
            """,
            (1, True),
            id="immortal-game",
        ),
        pytest.param(
            """
            1.e4 e5 2.Nf3 d6 3.d4 Bg4 4.dxe5 Bxf3 5.Qxf3 dxe5 6.Bc4 Nf6 7.Qb3 Qe7 8.Nc3
            c6 9.Bg5 b5 10.Nxb5 cxb5 11.Bxb5+ Nbd7 12.O-O-O Rd8 13.Rxd7 Rxd7 14.Rd1 Qe6 15.Bxd7+ Nxd7
            16.Qb8+ Nxb8 17.Rd8# 1-0
            """,
            (1, True),
            id="opera-game",
        ),
        pytest.param(
            """
            1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Nxe4 5. d4 Nd6 6. dxe5 Nxb5 7. a4 Nbd4 8. Nxd4 Nxd4 9. Qxd4 d5 10. exd6 Qxd6 11. Qe4+ Qe6 12. Qd4 Qd6 13. Qe4+ Qe6 14. Qd4 *
            """,
            (0, True),
            id="berlin-draw",
        ),
    ]
)
def test_repr_roundtrip_realgames(pgn_str: str, expected_outcome: tuple[int, bool]) -> None:
    pgn = io.StringIO(pgn_str.strip())
    game = chess.pgn.read_game(pgn)
    assert game is not None

    chess_game = ChessGame()

    enc1 = chess_game.encode_for_model()
    compact = chess_game.state.to_compact_repr(chess_game.current_player)
    enc2 = compact.unpack()
    assert np.array_equal(enc1, enc2)

    for move in game.mainline_moves():
        assert chess_game.state.get_value_and_terminated() == (0, False), chess_game.state.board
        action = chess_game.parse_move_str(move.uci())
        chess_game.make_move(action)

        enc1 = chess_game.encode_for_model()
        compact = chess_game.state.to_compact_repr(chess_game.current_player)
        enc2 = compact.unpack()
        assert np.array_equal(enc1, enc2)

    assert chess_game.state.get_value_and_terminated() == expected_outcome
