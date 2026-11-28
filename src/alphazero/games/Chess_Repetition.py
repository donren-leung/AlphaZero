from collections import Counter
from dataclasses import dataclass, field
from struct import Struct
from typing import Hashable

import bulletchess as chess
import numpy as np
from numpy.typing import NDArray

from .ChessUtils import TIME_STEPS, _frame_from_masks, history_frame, _piece_masks

# Seven uint64 bitboards: six piece types plus White occupancy.
# Three uint8 fields: side to move, castling rights, and legal EP square (64 = none).
_KEY = Struct("<7QBBB")
_CASTLES = (chess.WHITE_KINGSIDE, chess.WHITE_QUEENSIDE, chess.BLACK_KINGSIDE, chess.BLACK_QUEENSIDE)
# Map each of the 2^4 = 16 possible castling rights to a four-bit mask for the repetition key.
_CASTLING_MASK = {
    chess.CastlingRights([right for i, right in enumerate(_CASTLES) if mask & (1 << i)]): mask
    for mask in range(16)
}
# Sentinel for "no en passant square
NO_EP = 64

def legal_ep_square(board: chess.Board) -> chess.Square | None:
    ep = board.en_passant_square
    if ep is not None:
        # A double pawn push sets the native FEN target even if nobody can capture.
        # See if any pawn of the current player can capture it, if so ONLY THEN
        # test if it would be a legal move.
        pawns = board[board.turn, chess.PAWN]
        attackers = (ep.sw(), ep.se()) if board.turn == chess.WHITE else (ep.nw(), ep.ne())
        if any(sq is not None and sq in pawns for sq in attackers):
            if any(m.destination == ep and m.origin in pawns for m in board.legal_moves()):
                return ep
    return None

def position_key(board: chess.Board) -> bytes:
    """
    Collision-free repetition identity. Move counters and inert EP rights are excluded.
    """
    ep = legal_ep_square(board)
    return _KEY.pack(
        # 7Q
        *(int(board[p]) for p in chess.PIECE_TYPES),
        int(board[chess.WHITE]),
        # 3 bytes
        board.turn == chess.WHITE,
        _CASTLING_MASK[board.castling_rights],
        ep.index() if ep else NO_EP
    )

def compact_fen(board: chess.Board) -> str:
    fields = board.fen().split()
    if legal_ep_square(board) is None:
        fields[3] = "-"
    return " ".join(fields)

@dataclass(slots=True)
class CachedPosition:
    key: bytes
    history: tuple[NDArray[np.uint8], ...] # Newest first, fixed White perspective.
    children: dict[chess.Move, "CachedPosition"] = field(default_factory=dict)

class RepetitionTracker:
    def __init__(self, board: chess.Board):
        self.board = board
        self.key_counts: Counter[Hashable] = Counter()
        self.cache_stack: list[CachedPosition] = []

        # Counts for reconstructing the history planes, including positions
        # before the most recent irreversible move if a frame needs them.
        history_counts: Counter[bytes] = Counter()
        # Save the piece masks for the TIME_STEPS most recent positions.
        recent: list[tuple[bytes, NDArray[np.uint64]]] = []

        remaining_prior_moves = len(board.history)
        cursor = board.copy() if remaining_prior_moves else board
        in_current_segment = True

        # We must rewind the board for at least TIME_STEPS
        # and then until the next irreversible move
        while True:
            key = position_key(cursor)

            history_counts[key] += 1
            if in_current_segment:
                self.key_counts[key] += 1

            if len(recent) < TIME_STEPS:
                recent.append((key, _piece_masks(cursor)))

            if remaining_prior_moves == 0:
                break

            # Pawn moves and captures reset the clock in the resulting position.
            # Castling rights and an expiring EP capture require the prior position to check.
            zeroing_move = cursor.halfmove_clock == 0
            rights_after = cursor.castling_rights
            cursor.undo()
            remaining_prior_moves -= 1

            irreversible = (
                # Move that was undone was a pawn move or capture or a move that removed castling.
                zeroing_move
                or cursor.castling_rights != rights_after
                # Regardless of whether EP was chosen, any subsequent position
                # cannot have the same EP square i.e. the undone move was irreversible.
                or legal_ep_square(cursor) is not None
            )

            if irreversible:
                # Don't count any further historic positions for the live game's count.
                in_current_segment = False
                # If we have enough history, stop here.
                # Otherwise, we have to keep going until the next irreversible move,
                # even if we rewind more than TIME_STEPS, so we can fully reconstruct
                # the history planes (repetition counts).
                if len(recent) == TIME_STEPS:
                    break

        # Recent is newest first. Decrement each visit when travelling back in time
        # older frames never count an occurrence that is in its future.
        frames = []
        for key, masks in recent:
            frames.append(_frame_from_masks(masks, history_counts[key]))
            history_counts[key] -= 1

        self.num_repeated = sum(count >= 2 for count in self.key_counts.values())
        current_key = recent[0][0]
        self.curr_cache = CachedPosition(current_key, tuple(frames))

    def push(self, move: chess.Move) -> None:
        self.board.apply(move)
        parent = self.curr_cache
        child = parent.children.get(move)

        if child is None:
            # Previously untraversed move
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
        self.push(move) # self.curr_cache is now the child

        # A search copy may still own the old root. Discard its unplayed siblings.
        parent.children = {move: self.curr_cache}
        self.cache_stack.clear()
        if self.board.halfmove_clock == 0 or rights != self.board.castling_rights or ep is not None:
            self.key_counts = Counter({self.curr_cache.key: 1})
            self.num_repeated = 0

    def is_threefold(self) -> bool:
        return self.key_counts[self.curr_cache.key] >= 3

    def can_claim_threefold(self) -> bool:
        # Current position already occurred three times.
        if self.is_threefold():
            return True

        # Without any repeated position in the history,
        # no legal move can produce a third occurrence.
        if self.num_repeated == 0:
            return False

        # Only when repetitions exist do we need to
        # investigate next-move claims.
        for move in self.board.legal_moves():
            child = self.curr_cache.children.get(move)

            # Get key of position after move
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
        clone.num_repeated = self.num_repeated

        clone.curr_cache = self.curr_cache
        clone.cache_stack = self.cache_stack.copy()

        return clone
