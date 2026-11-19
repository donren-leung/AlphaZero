from dataclasses import dataclass
from typing import ClassVar

import chess
import numpy as np
from numpy.typing import NDArray

from .GameStateBase import GameStateBase
from .GameBase import GameBase
from .ChessUtils import encode_move, decode_action, encode_chess, history_frame, RepetitionTracker, CHESS_ACTION_SIZE, CHESS_INPUT_CHANNELS

type Board = chess.Board
type ModelInput = NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class CompactChessState:
    # FEN is the position up to 7 half-moves back
    # Zip of the following:
    # Repetitions: Oldest first. Counts occurences of position at t (before move t).
    # Moves: Oldest first. Populates a partial history of the game. One elem shorter than repetitions.

    fen: str
    repetition_flags: tuple[int, ...]
    moves: tuple[str, ...]

    def unpack(self) -> ModelInput:
        board = chess.Board(self.fen)
        history_frames: list[NDArray[np.uint8]] = [history_frame(board, self.repetition_flags[0])]
        for uci, repetitions in zip(self.moves, self.repetition_flags[1:], strict=True):
            board.push_uci(uci)
            history_frames.append(history_frame(board, repetitions))

        # encode_chess expects reverse (newest first) order
        return encode_chess(board, history_frames[::-1])

class ChessState(GameStateBase[chess.Move]):
    row_count: ClassVar[int] = 8
    col_count: ClassVar[int] = 8

    __slots__ = ["board", "repetition_tracker"]
    def __init__(self, board: Board, repetition_tracker: RepetitionTracker | None = None):
        self.board = board
        self.repetition_tracker = repetition_tracker or RepetitionTracker(board)

    def get_legal_actions(self) -> NDArray[np.bool_]:
        player = 1 if self.board.turn == chess.WHITE else -1
        legal_actions = np.zeros(CHESS_ACTION_SIZE, dtype=np.bool_)
        for move in self.board.legal_moves:
            action = encode_move(self.board, move, player)
            legal_actions[action] = True
        return legal_actions

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'ChessState':
        move = decode_action(self.board, action, player)
        if copy:
            new_state = ChessState(self.board.copy())
            new_state.push_to_board(move)
            return new_state
        else:
            irreversible = self.board.is_irreversible(move)
            self.repetition_tracker.purge_cache_except(move, irreversible)
            self.push_to_board(move)
            self.repetition_tracker.cache_stack.clear()
            return self

    def encode_move(self, move: chess.Move) -> int:
        return encode_move(self.board, move, 1 if self.board.turn == chess.WHITE else -1)

    def decode_action(self, action: int, player: int) -> chess.Move:
        return decode_action(self.board, action, player)

    def push_to_board(self, move: chess.Move) -> None:
        self.repetition_tracker.push(move)

    def move_repr(self, move: chess.Move) -> str:
        return self.board.san(move)

    def pop_board(self) -> chess.Move:
        return self.repetition_tracker.pop()

    def get_value_and_terminated(self):
        if self.board.ply() >= 512:
            # Hard cap on 512 half-moves.
            return 0, True

        outcome = self.board.outcome(claim_draw=False)
        if outcome is not None:
            # Normal game end
            if outcome.winner is None:
                # Draw (by stalemate or insufficient material)
                return 0, True
            # Checkmate. The player who just moved is the winner, so the current player is the loser.
            assert outcome.winner is not self.board.turn
            return 1, True
        else:
            # Potentially draw by claimable 50-move rule or 3-fold repetition
            # Or game is still ongoing.
            if self.board.can_claim_fifty_moves():
                return 0, True
            if self.repetition_tracker.can_claim_threefold():
                return 0, True
            return 0, False

    def encode_for_model(self, player: int=0) -> ModelInput:
        return encode_chess(self.board, self.repetition_tracker.curr_cache.history)

    def to_compact_repr(self, player: int=0) -> CompactChessState:
        board = self.board.copy(stack=True)

        moves: list[str] = []
        repetition_flags: list[int] = []

        for i in range(8):
            # Total occurrences including this position, capped at 3.
            if board.is_repetition(3):
                count = 3
            elif board.is_repetition(2):
                count = 2
            else:
                count = 1

            repetition_flags.append(count)

            if i == 7 or not board.move_stack:
                # Need N=8 repetition counts but only N-1 moves
                # + starting position gets 8 positions.
                break

            move = board.pop()
            moves.append(move.uci())

        assert len(repetition_flags) == len(moves) + 1
        return CompactChessState(
            fen=board.fen(),
            repetition_flags=tuple(reversed(repetition_flags)),
            moves=tuple(reversed(moves)),
        )

    def __str__(self) -> str:
        return str(self.board)

    def __copy__(self) -> 'ChessState':
        board = self.board.copy()
        tracker = self.repetition_tracker.clone_for_board(board)
        return ChessState(board, tracker)

class ChessGame(GameBase[ChessState]):
    action_size: ClassVar[int] = CHESS_ACTION_SIZE
    input_channels: ClassVar[int] = CHESS_INPUT_CHANNELS
    row_count: ClassVar[int] = 8
    col_count: ClassVar[int] = 8

    def __init__(self, state: ChessState | None=None) -> None:
        super().__init__(state)

    @classmethod
    def get_initial_state(cls) -> ChessState:
        return ChessState(chess.Board())

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return self.state.get_legal_actions()

    def _make_move(self, action: int, player: int) -> None:
        self.state = self.state.get_next_state(action, player, copy=False)

    def get_value_and_terminated(self):
        return self.state.get_value_and_terminated()

    def encode_for_model(self) -> ModelInput:
        return self.state.encode_for_model()

    def parse_move_str(self, cmd: str) -> int:
        move = self.state.board.parse_san(cmd)
        action = encode_move(self.state.board, move, self.current_player)

        return action

    def get_legal_moves(self, legal_actions: NDArray[np.bool_]) -> list[str]:
        """
        Print the legal moves in a human-readable format (e.g. "e4" in chess).
        """
        return [
            self.state.board.san(decode_action(self.state.board, i, self.current_player))
            for i in range(self.action_size) if legal_actions[i]
        ]

    def __repr__(self) -> str:
        return "Chess"

    def __str__(self) -> str:
        return self.state.__str__()
