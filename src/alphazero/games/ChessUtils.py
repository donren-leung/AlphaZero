from collections import Counter, deque
from dataclasses import dataclass, field
from struct import Struct
from typing import Sequence

import bulletchess as chess
import numpy as np
from numpy.typing import NDArray

CHESS_ENCODING_VERSION = 1
CHESS_INPUT_VERSION = 1
CHESS_ACTION_SIZE = 8 * 8 * 73
CHESS_INPUT_CHANNELS = 119
TIME_STEP_LAYERS = 14
CHESS_PIECE_LAYERS = 6
REPETITION_PLANE_START = 12
TIME_STEPS = 8
AUXILIARY_PLANE_START = TIME_STEP_LAYERS * TIME_STEPS

QUEEN_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
KNIGHT_OFFSETS = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


type ActionTables = tuple[dict[chess.Move, int], list[chess.Move | None], dict[int, chess.Move | None]]


def _action_tables(black: bool) -> ActionTables:
    """Index v1 policies file-first, with ranks mirrored for Black."""
    planes: list[tuple[int, int, chess.PieceType | None]] = [
        (distance * df, distance * dr, None)
        for distance in range(1, 8) for df, dr in QUEEN_DIRECTIONS
    ]
    planes.extend((df, dr, None) for df, dr in KNIGHT_OFFSETS)
    planes.extend((df, 1, piece) for df in (-1, 0, 1) for piece in UNDERPROMOTION_PIECES)
    by_move: dict[chess.Move, int] = {}
    by_action: list[chess.Move | None] = [None] * CHESS_ACTION_SIZE
    queen_promotions: dict[int, chess.Move | None] = {}
    flip = 56 if black else 0
    for position in range(64):
        file, rank = divmod(position, 8)
        source = chess.SQUARES[(rank * 8 + file) ^ flip]
        for plane, (df, dr, promotion) in enumerate(planes):
            f, r = file + df, rank + dr
            if not (0 <= f < 8 and 0 <= r < 8):
                continue
            if promotion is not None and r != 7:
                continue
            target = chess.SQUARES[(r * 8 + f) ^ flip]
            action = position * 73 + plane
            move = chess.Move(source, target, promotion)
            by_action[action] = move
            by_move[move] = action
            if promotion is None and r == 7:
                queen_promotions[action] = None
                if rank == 6 and abs(df) <= 1:
                    queen = chess.Move(source, target, chess.QUEEN)
                    by_move[queen] = action
                    queen_promotions[action] = queen
    return by_move, by_action, queen_promotions


_ACTIONS = {chess.WHITE: _action_tables(False), chess.BLACK: _action_tables(True)}


def _check_player(board: chess.Board, player: int) -> None:
    assert type(player) is int and player in (1, -1), "player must be 1 (White) or -1 (Black)"
    assert (board.turn == chess.WHITE) == (player == 1), "player must match board.turn"


def legal_mask(board: chess.Board) -> NDArray[np.bool_]:
    actions = _ACTIONS[board.turn][0]
    mask = np.zeros(CHESS_ACTION_SIZE, dtype=np.bool_)
    for move in board.legal_moves():
        mask[actions[move]] = True
    return mask


def encode_move(board: chess.Board, move: chess.Move, player: int) -> int:
    _check_player(board, player)
    try:
        return _ACTIONS[board.turn][0][move]
    except KeyError as exc:
        raise ValueError(f"Unrepresentable move {move}") from exc


def decode_action(board: chess.Board, action: int, player: int) -> chess.Move:
    _check_player(board, player)
    if type(action) is not int or not 0 <= action < CHESS_ACTION_SIZE:
        raise ValueError(f"action must be an integer in [0, {CHESS_ACTION_SIZE}), got {action=}")
    _, moves, queens = _ACTIONS[board.turn]
    move = moves[action]
    if move is None:
        raise ValueError(f"Action {action} points off the board or encodes an impossible promotion")
    if action in queens:
        piece = board[move.origin]
        if piece is not None and piece.piece_type == chess.PAWN:
            queen = queens[action]
            if queen is None:
                raise ValueError(f"Action {action} encodes an impossible promotion")
            return queen
    return move


def board_ply(board: chess.Board) -> int:
    return 2 * (board.fullmove_number - 1) + (board.turn == chess.BLACK)


def encode_chess(board: chess.Board, history: Sequence[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    features = np.zeros((CHESS_INPUT_CHANNELS, 8, 8), dtype=np.uint8)
    turn = board.turn
    for t, frame in enumerate(history[:TIME_STEPS]):
        dst = features[t * TIME_STEP_LAYERS:(t + 1) * TIME_STEP_LAYERS]
        if turn == chess.WHITE:
            dst[:] = frame
        else:
            dst[:6] = frame[6:12, ::-1, :]
            dst[6:12] = frame[:6, ::-1, :]
            dst[12:14] = frame[12:14]

    aux = features[AUXILIARY_PLANE_START:]
    rights = board.castling_rights
    for side, color in enumerate((turn, turn.opposite)):
        aux[2 * side].fill(rights.kingside(color))
        aux[2 * side + 1].fill(rights.queenside(color))
    aux[4, 0, :2] = divmod(board_ply(board), 256)
    aux[5].fill(board.halfmove_clock)
    return features


_PIECES = tuple(chess.Piece(c, p) for c in (chess.WHITE, chess.BLACK) for p in chess.PIECE_TYPES)


def history_frame(board: chess.Board, occurrences: int) -> NDArray[np.uint8]:
    frame = np.empty((TIME_STEP_LAYERS, 8, 8), dtype=np.uint8)
    masks = np.array([int(board[piece]) for piece in _PIECES], dtype="<u8")
    frame[:12] = np.unpackbits(masks.view(np.uint8), bitorder="little").reshape(12, 8, 8)
    frame[12].fill(occurrences >= 2)
    frame[13].fill(occurrences >= 3)
    frame.flags.writeable = False
    return frame


def legal_ep_square(board: chess.Board) -> chess.Square | None:
    ep = board.en_passant_square
    if ep is not None:
        # A double pawn push sets the native FEN target even if nobody can capture.
        pawns = board[board.turn, chess.PAWN]
        attackers = (ep.sw(), ep.se()) if board.turn == chess.WHITE else (ep.nw(), ep.ne())
        if any(sq is not None and sq in pawns for sq in attackers):
            if any(m.destination == ep and m.origin in pawns for m in board.legal_moves()):
                return ep
    return None


_KEY = Struct("<7QBBB")
_CASTLES = (chess.WHITE_KINGSIDE, chess.WHITE_QUEENSIDE, chess.BLACK_KINGSIDE, chess.BLACK_QUEENSIDE)
_CASTLING_MASK = {
    chess.CastlingRights([right for i, right in enumerate(_CASTLES) if mask & (1 << i)]): mask
    for mask in range(16)
}


def position_key(board: chess.Board) -> bytes:
    """Collision-free repetition identity. Move counters and inert EP rights are excluded."""
    ep = legal_ep_square(board)
    return _KEY.pack(*(int(board[p]) for p in chess.PIECE_TYPES), int(board[chess.WHITE]),
                     board.turn == chess.WHITE, _CASTLING_MASK[board.castling_rights], ep.index() if ep else 64)


def compact_fen(board: chess.Board) -> str:
    fields = board.fen().split()
    if legal_ep_square(board) is None:
        fields[3] = "-"
    return " ".join(fields)


@dataclass(slots=True)
class CachedPosition:
    key: bytes
    history: tuple[NDArray[np.uint8], ...]  # Newest first, fixed White perspective.
    children: dict[chess.Move, "CachedPosition"] = field(default_factory=dict)


class RepetitionTracker:
    def __init__(self, board: chess.Board):
        self.board = board
        self.key_counts: Counter[bytes] = Counter()
        self.cache_stack: list[CachedPosition] = []
        self.num_repeated = 0
        # Rebuild counts from the whole path, but materialize only its last eight frames.
        moves = board.history
        root = board.copy() if moves else board
        for _ in moves:
            root.undo()
        frames: deque[NDArray[np.uint8]] = deque(maxlen=TIME_STEPS)
        for ply in range(len(moves) + 1):
            if ply:
                root.apply(moves[ply - 1])
                if root.halfmove_clock == 0:
                    self.key_counts.clear()
                    self.num_repeated = 0
            key = position_key(root)
            self.key_counts[key] += 1
            if self.key_counts[key] == 2:
                self.num_repeated += 1
            if len(moves) - ply < TIME_STEPS:
                frames.appendleft(history_frame(root, self.key_counts[key]))
        self.curr_cache = CachedPosition(key, tuple(frames))

    def push(self, move: chess.Move) -> None:
        self.board.apply(move)
        parent = self.curr_cache
        child = parent.children.get(move)
        if child is None:
            key = position_key(self.board)
            frame = history_frame(self.board, self.key_counts[key] + 1)
            child = CachedPosition(key, (frame, *parent.history[:TIME_STEPS - 1]))
            parent.children[move] = child
        self.key_counts[child.key] += 1
        if self.key_counts[child.key] == 2:
            self.num_repeated += 1
        self.cache_stack.append(parent)
        self.curr_cache = child

    def pop(self) -> chess.Move:
        if not self.cache_stack:
            raise IndexError("Cannot pop before the tracker's initial position")
        key = self.curr_cache.key
        self.key_counts[key] -= 1
        if self.key_counts[key] == 1:
            self.num_repeated -= 1
        elif self.key_counts[key] == 0:
            del self.key_counts[key]
        self.curr_cache = self.cache_stack.pop()
        move = self.board.undo()
        assert move is not None
        return move

    def commit(self, move: chess.Move) -> None:
        parent = self.curr_cache
        rights = self.board.castling_rights
        ep = legal_ep_square(self.board)
        self.push(move)
        # A search copy may still own the old root. Discard its unplayed siblings.
        parent.children = {move: self.curr_cache}
        self.cache_stack.clear()
        if self.board.halfmove_clock == 0 or rights != self.board.castling_rights or ep is not None:
            self.key_counts = Counter({self.curr_cache.key: 1})
            self.num_repeated = 0

    def can_claim_threefold(self) -> bool:
        if self.key_counts[self.curr_cache.key] >= 3:
            return True
        if self.num_repeated:
            for move in self.board.legal_moves():
                child = self.curr_cache.children.get(move)
                if child is None:
                    self.board.apply(move)
                    try:
                        key = position_key(self.board)
                    finally:
                        self.board.undo()
                else:
                    key = child.key
                if self.key_counts[key] >= 2:
                    return True
        return False

    def clone_for_board(self, board: chess.Board) -> "RepetitionTracker":
        clone = object.__new__(type(self))
        clone.board = board
        clone.key_counts = self.key_counts.copy()
        clone.curr_cache = self.curr_cache
        clone.cache_stack = self.cache_stack.copy()
        clone.num_repeated = self.num_repeated
        return clone
