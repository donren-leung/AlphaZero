import io
import random

import bulletchess
import chess
import chess.pgn
import numpy as np
import pytest

from alphazero.games.Chess import ChessGame, ChessState
from alphazero.games.ChessUtils import CHESS_ACTION_SIZE, CHESS_ENCODING_VERSION, decode_action, encode_move
from chess_reference import native_board, old_encode_chess, reference_action, reference_decode


def assert_parity(reference: chess.Board, state: ChessState):
    player = 1 if reference.turn else -1
    expected = {reference_action(reference, m, player): m.uci() for m in reference.legal_moves}
    actual = {state.encode_move(m): m.uci() for m in state.board.legal_moves()}
    assert expected == actual
    mask = state.get_legal_actions()
    assert mask.shape == (4672,) and mask.dtype == np.bool_
    assert np.flatnonzero(mask).tolist() == sorted(expected)
    for action, uci in expected.items():
        assert state.decode_action(action, player).uci() == uci
    encoded = old_encode_chess(reference)
    for tensor in (state.encode_for_model(), state.to_compact_repr().unpack()):
        assert tensor.dtype == np.uint8
        np.testing.assert_array_equal(tensor, encoded)
    outcome = reference.outcome(claim_draw=True)
    expected_result = (0, True) if reference.ply() >= 512 else (
        int(outcome is not None and outcome.winner is not None), outcome is not None
    )
    assert state.get_value_and_terminated() == expected_result


def test_constants():
    assert CHESS_ACTION_SIZE == 4672
    assert CHESS_ENCODING_VERSION == 1

@pytest.mark.parametrize("seed", [7, 42, 91])
def test_initial_and_random_play(seed):
    reference = chess.Board()
    state = ChessState(bulletchess.Board())
    rng = random.Random(seed)
    for _ in range(120):
        assert_parity(reference, state)
        if reference.is_game_over(claim_draw=True):
            break
        move = rng.choice(list(reference.legal_moves))
        state.get_next_state(reference_action(reference, move, 1 if reference.turn else -1),
                             1 if reference.turn else -1, copy=False)
        reference.push(move)

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
    reference = chess.Board(fen)
    state = ChessState(native_board(reference))
    assert_parity(reference, state)
    game = ChessGame(state)
    for move in reference.legal_moves:
        native = bulletchess.Move.from_uci(move.uci())
        san = reference.san(move)
        assert state.move_repr(native) == san
        assert game.parse_move_str(san) == game.parse_move_str(move.uci()) == state.encode_move(native)


def test_invalid():
    board = bulletchess.Board()
    with pytest.raises(AssertionError):
        encode_move(board, bulletchess.Move.from_uci("e2e4"), -1)
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

    white = native_board(white_board)
    black = native_board(black_board)
    for white_move in white_board.legal_moves:
        black_move = chess.Move(
            chess.square_mirror(white_move.from_square),
            chess.square_mirror(white_move.to_square),
            promotion=white_move.promotion,
        )
        assert black_move in black_board.legal_moves
        action = encode_move(white, bulletchess.Move.from_uci(white_move.uci()), 1)
        assert encode_move(black, bulletchess.Move.from_uci(black_move.uci()), -1) == action
        assert decode_action(black, action, -1).uci() == black_move.uci()


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
    """Compare every slot, including off-board geometry and impossible promotions."""
    reference = chess.Board(fen)
    board = native_board(reference)
    player = 1 if reference.turn else -1
    for action in range(CHESS_ACTION_SIZE):
        try:
            expected = reference_decode(reference, action, player)
        except ValueError:
            with pytest.raises(ValueError, match="off the board"):
                decode_action(board, action, player)
            continue

        if expected.promotion:
            # python-chess can describe an impossible promotion, even from the second rank.
            rank = chess.square_rank(expected.from_square)
            df = chess.square_file(expected.to_square) - chess.square_file(expected.from_square)
            if rank != (6 if reference.turn else 1) or abs(df) > 1:
                assert expected not in reference.legal_moves
                with pytest.raises(ValueError, match="impossible promotion"):
                    decode_action(board, action, player)
                continue
        assert decode_action(board, action, player).uci() == expected.uci()


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

    state = ChessState(native_board(board))
    assert_parity(board, state)
    for move in game.mainline_moves():
        state.push_to_board(bulletchess.Move.from_uci(move.uci()))
        board.push(move)
        assert_parity(board, state)

    assert board.outcome(claim_draw=True) is not None


@pytest.mark.parametrize("fen", [
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",   # pinned EP capturer
    "7k/8/8/8/8/8/8/R3K3 w - - 100 51",
    "7k/8/8/8/8/8/8/R3K3 w - - 150 76",
    "7k/6Q1/5K2/8/8/8/8/8 b - - 150 76",    # mate precedes clock draws
    "7k/8/8/8/8/8/8/R3K3 b - - 2 256",       # ply 511
    "7k/8/8/8/8/8/8/R3K3 w - - 2 257",       # ply 512
    "8/8/8/8/8/5k2/6q1/7K w - - 0 257",      # cap precedes mate
    "2b4k/8/4b3/8/2B5/8/8/K7 w - - 0 1",    # same-color bishops
    "2b4k/8/8/8/3B4/8/8/K7 w - - 0 1",      # opposite-color bishops
])
def test_rule_boundaries(fen):
    board = chess.Board(fen)
    assert_parity(board, ChessState(native_board(board)))


def test_search_undo_siblings_and_commit():
    reference = chess.Board()
    state = ChessState(native_board(reference))
    for san in ("Nf3", "Nf6", "Ng1", "Ng8") * 4:
        assert_parity(reference, state)
        move = reference.push_san(san)
        state.push_to_board(bulletchess.Move.from_uci(move.uci()))
    assert reference.is_fivefold_repetition()
    assert_parity(reference, state)
    for _ in range(9):
        assert state.pop_board().uci() == reference.pop().uci()
        assert_parity(reference, state)
    # Seven plies in, Black can announce Ng8 and claim the third occurrence.
    assert reference.can_claim_threefold_repetition() and not reference.is_repetition(3)
    for san in ("e5", "Ng8", "e5"):
        move = reference.push_san(san)
        state.push_to_board(bulletchess.Move.from_uci(move.uci()))
        assert_parity(reference, state)
        state.pop_board()
        reference.pop()
    copied = state.__copy__()
    move = reference.parse_san("e5")
    state.get_next_state(reference_action(reference, move, -1), -1, copy=False)
    assert_parity(reference, copied)
    reference.push(move)
    assert_parity(reference, state)
    with pytest.raises(IndexError):
        state.pop_board()


@pytest.mark.parametrize(("fen", "field", "replacement", "same"), [
    ("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 3, "-", True),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 3, "-", False),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 2, "Kkq", False),
    (chess.STARTING_FEN, 4, "12", True),
    (chess.STARTING_FEN, 1, "b", False),
])
def test_repetition_identity_and_compact_fen(fen, field, replacement, same):
    from alphazero.games.ChessUtils import position_key
    fields = fen.split()
    fields[field] = replacement
    board = bulletchess.Board.from_fen(fen)
    assert (position_key(board) == position_key(bulletchess.Board.from_fen(" ".join(fields)))) == same
    assert ChessState(board).to_compact_repr().fen == chess.Board(fen).fen()
