from dataclasses import dataclass
from re import fullmatch, sub
from typing import ClassVar

import bulletchess as chess
import numpy as np
from numpy.typing import NDArray

from .GameStateBase import GameStateBase
from .GameBase import GameBase
from .ChessUtils import (
    CHESS_ACTION_SIZE, CHESS_INPUT_CHANNELS,
    encode_move, decode_action, encode_chess,
    board_ply, legal_mask, history_frame
)
from .Chess_Repetition import RepetitionTracker, compact_fen

type Board = chess.Board
type ModelInput = NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class CompactChessState:
    """
    FEN is the position up to 7 half-moves back
    Zip of the following:
    Repetitions: Oldest first. Counts occurences of position at t (before move t).
    Moves: Oldest first. Populates a partial history of the game. One elem shorter than repetitions.
    """

    fen: str
    repetition_flags: tuple[int, ...]
    moves: tuple[str, ...]

    def unpack(self) -> ModelInput:
        board = chess.Board.from_fen(self.fen)
        history_frames: list[NDArray[np.uint8]] = [history_frame(board, self.repetition_flags[0])]
        for uci, repetitions in zip(self.moves, self.repetition_flags[1:], strict=True):
            move = chess.Move.from_uci(uci)
            if move is None or move not in board.legal_moves():
                raise ValueError(f"Illegal compact replay move {uci}")
            board.apply(move)
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
        return legal_mask(self.board)

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'ChessState':
        move = decode_action(self.board, action, player)
        if copy:
            new_state = self.__copy__()
            new_state.repetition_tracker.cache_stack.clear()
            new_state.push_to_board(move)
            return new_state
        else:
            self.repetition_tracker.commit(move)
            return self

    def encode_move(self, move: chess.Move) -> int:
        return encode_move(self.board, move, 1 if self.board.turn == chess.WHITE else -1)

    def decode_action(self, action: int, player: int) -> chess.Move:
        return decode_action(self.board, action, player)

    def push_to_board(self, move: chess.Move) -> None:
        self.repetition_tracker.push(move)

    def move_repr(self, move: chess.Move) -> str:
        san = move.san(self.board)
        # bulletchess 0.4 formats promotions as e8Q. The CLI uses canonical e8=Q.
        if move.promotion is not None:
            return sub(r"([18])([QRBN])", r"\1=\2", san)
        return san

    def pop_board(self) -> chess.Move:
        return self.repetition_tracker.pop()

    def get_value_and_terminated(self):
        board = self.board
        if board_ply(board) >= 512:
            # Hard cap on 512 half-moves.
            return 0, True

        if board in chess.CHECKMATE:
            return 1, True
        if board in chess.STALEMATE or board in chess.INSUFFICIENT_MATERIAL:
            return 0, True
        if board.halfmove_clock >= 100:
            return 0, True
        if board.halfmove_clock == 99:
            # A player may announce a non-zeroing move that reaches 100 halfmoves.
            for move in board.legal_moves():
                board.apply(move)
                try:
                    if board.halfmove_clock == 100 and board not in chess.MATE:
                        return 0, True
                finally:
                    board.undo()
        return 0, self.repetition_tracker.can_claim_threefold()

    def encode_for_model(self, player: int=0) -> ModelInput:
        return encode_chess(self.board, self.repetition_tracker.curr_cache.history)

    def to_compact_repr(self, player: int=0) -> CompactChessState:
        board = self.board.copy()
        frames = self.repetition_tracker.curr_cache.history
        repetitions = tuple(1 + int(f[12, 0, 0]) + int(f[13, 0, 0]) for f in reversed(frames))
        moves = []
        for _ in range(len(frames) - 1):
            move = board.undo()
            assert move is not None
            moves.append(move.uci())

        assert len(repetitions) == len(moves) + 1
        return CompactChessState(compact_fen(board), repetitions, tuple(reversed(moves)))

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
        super().__init__(state,  starting_player=-1 if state is not None and state.board.turn == chess.BLACK else 1)

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
        board = self.state.board
        # Accept either UCI or SAN.
        if fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", cmd):
            move = chess.Move.from_uci(cmd)
        else:
            try:
                move = chess.Move.from_san(cmd.replace("0", "O"), board)
            except ValueError:
                raise ValueError(f"Invalid SAN move: {cmd}") from None
        if move is None or move not in board.legal_moves():
            raise ValueError(f"Illegal move: {cmd}")
        return encode_move(board, move, self.current_player)

    def get_legal_moves(self, legal_actions: NDArray[np.bool_]) -> list[str]:
        """
        Print the legal moves in a human-readable format (e.g. "e4" in chess).
        """
        return [
            self.state.move_repr(
                self.state.decode_action(int(i), self.current_player)
            ) for i in np.flatnonzero(legal_actions)
        ]

    def __repr__(self) -> str:
        return "Chess"

    def __str__(self) -> str:
        return self.state.__str__()
