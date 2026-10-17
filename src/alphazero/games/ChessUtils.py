
import chess

CHESS_ENCODING_VERSION = 1
CHESS_ACTION_SIZE = 8 * 8 * 73

# (file delta, rank delta) in white-facing coordinates, clockwise from north
QUEEN_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
KNIGHT_OFFSETS = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
UNDERPROMOTION_DIRECTIONS = ((-1, 1), (0, 1), (1, 1))
UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)

def _check_player(board: chess.Board, player: int) -> None:
    assert type(player) is int and player in (1, -1), "player must be 1 (White) or -1 (Black)"
    assert board.turn == (player == 1), "player must match board.turn"
    if board.chess960:
        raise ValueError("Chess960 is not supported by this standard-chess action codec")

def _neutral_square(square: chess.Square, player: int) -> chess.Square:
    return square if player == 1 else chess.square_mirror(square)

def _unneutral_square(square: chess.Square, player: int) -> chess.Square:
    return square if player == 1 else chess.square_mirror(square)

def encode_move(board: chess.Board, move: chess.Move, player: int) -> int:
    """Encode a legal standard-chess move as an integer in [0, 4672)."""
    _check_player(board, player)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for current board: {move.uci()}")

    source = _neutral_square(move.from_square, player)
    target = _neutral_square(move.to_square, player)
    df = chess.square_file(target) - chess.square_file(source)
    dr = chess.square_rank(target) - chess.square_rank(source)

    if move.promotion in UNDERPROMOTION_PIECES:
        try:
            direction = UNDERPROMOTION_DIRECTIONS.index((df, dr))
        except ValueError as exc:
            raise ValueError(f"Invalid underpromotion vector: {(df, dr)}") from exc
        plane = 64 + direction * 3 + UNDERPROMOTION_PIECES.index(move.promotion)
    elif (df, dr) in KNIGHT_OFFSETS:
        plane = 56 + KNIGHT_OFFSETS.index((df, dr))
    else:
        distance = max(abs(df), abs(dr))
        if not 1 <= distance <= 7 or not (
            # horizontal, vertical, or diagonal move
            df == 0 or dr == 0 or abs(df) == abs(dr)
        ):
            raise ValueError(f"Unrepresentable move vector: {(df, dr)}")
        step = (df // distance if df else 0, dr // distance if dr else 0)
        plane = (distance - 1) * 8 + QUEEN_DIRECTIONS.index(step)

    return (chess.square_file(source) * 8 + chess.square_rank(source)) * 73 + plane

def decode_action(board: chess.Board, action: int, player: int) -> chess.Move:
    """Decode an action, rejecting invalid, off-board, or illegal moves."""
    _check_player(board, player)
    if type(action) is not int or not 0 <= action < CHESS_ACTION_SIZE:
        raise ValueError(f"action must be an integer in [0, {CHESS_ACTION_SIZE})")

    position, plane = divmod(action, 73)
    source_file, source_rank = divmod(position, 8)
    source = chess.square(source_file, source_rank)
    promotion = None

    if plane < 56:
        distance, direction = divmod(plane, 8)
        df, dr = QUEEN_DIRECTIONS[direction]
        df *= distance + 1
        dr *= distance + 1
    elif plane < 64:
        df, dr = KNIGHT_OFFSETS[plane - 56]
    else:
        direction, piece = divmod(plane - 64, 3)
        df, dr = UNDERPROMOTION_DIRECTIONS[direction]
        promotion = UNDERPROMOTION_PIECES[piece]

    target_file, target_rank = source_file + df, source_rank + dr
    if not (0 <= target_file < 8 and 0 <= target_rank < 8):
        raise ValueError(f"Action {action} points off the board")

    target = chess.square(target_file, target_rank)
    from_sq = _unneutral_square(source, player)
    to_sq = _unneutral_square(target, player)

    if promotion is None:
        chess_piece = board.piece_at(from_sq)
        if (
            chess_piece is not None
            and chess_piece.piece_type == chess.PAWN
            and chess.square_rank(target) == 7
        ):
            promotion = chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promotion)
    if move not in set(board.generate_legal_moves()):
        raise ValueError(
            f"Action {action} is not a canonical legal move: {move.uci()}"
        )
    return move
