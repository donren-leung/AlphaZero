from dataclasses import dataclass
from typing import ClassVar

import chess
import numpy as np
from numpy.typing import NDArray

from .GameStateBase import GameStateBase
from .GameBase import GameBase
from .ChessUtils import encode_move, decode_action, encode_chess, CHESS_ACTION_SIZE, CHESS_INPUT_CHANNELS

type Board = chess.Board
type ModelInput = NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class CompactChessState:
    # FEN is the position up to 7 half-moves back
    # Zip of the following:
    # Moves: Oldest first. Populates a partial history of the game.
    # Repetitions. Newest first. counts occurences of position at step t

    fen: str
    moves: tuple[str, ...]
    repetition_flags: tuple[int, ...]

    def unpack(self) -> ModelInput:
        board = chess.Board(self.fen)
        for uci in self.moves:
            board.push_uci(uci)

        return encode_chess(board, self.repetition_flags)

class ChessState(GameStateBase[chess.Move]):
    __slots__ = ["board", "row_count", "col_count"]
    def __init__(self, board: Board):
        self.board = board
        self.row_count = 8
        self.col_count = 8

    def get_legal_actions(self) -> NDArray[np.bool_]:
        player = 1 if self.board.turn == chess.WHITE else -1
        legal_actions = np.zeros(CHESS_ACTION_SIZE, dtype=np.bool_)
        for move in self.board.legal_moves:
            action = encode_move(self.board, move, player)
            legal_actions[action] = True
        return legal_actions

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'ChessState':
        if copy:
            new_state = self.board.copy()
        else:
            new_state = self.board

        move = decode_action(new_state, action, player)
        new_state.push(move)
        return ChessState(new_state)

    def encode_move(self, move: chess.Move) -> int:
        return encode_move(self.board, move, 1 if self.board.turn == chess.WHITE else -1)

    def decode_action(self, action: int, player: int) -> chess.Move:
        return decode_action(self.board, action, player)

    def push_to_board(self, move: chess.Move) -> None:
        self.board.push(move)

    def move_repr(self, move: chess.Move) -> str:
        return self.board.san(move)

    def pop_board(self) -> chess.Move:
        return self.board.pop()

    def get_value_and_terminated(self):
        if self.board.ply() >= 512:
            # Hard cap on 512 half-moves.
            return 0, True
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return 0, False
        if outcome.winner is None:
            return 0, True
        # Checkmate. The player who just moved is the winner, so the current player is the loser.
        assert outcome.winner is not self.board.turn
        return 1, True

    def encode_for_model(self, player: int=0) -> ModelInput:
        return encode_chess(self.board)

    def to_compact_repr(self, player: int=0) -> CompactChessState:
        board = self.board.copy(stack=True)

        moves: list[str] = []
        repetition_flags: list[int] = []

        for i in range(8):
            # Number of previous occurrences, capped at 2
            if board.is_repetition(3):
                count = 2
            elif board.is_repetition(2):
                count = 1
            else:
                count = 0

            repetition_flags.append(count)

            if i == 7 or not board.move_stack:
                break

            move = board.pop()
            moves.append(move.uci())

        return CompactChessState(
            fen=board.fen(),
            moves=tuple(reversed(moves)),
            repetition_flags=tuple(repetition_flags)
        )

    def __str__(self) -> str:
        return str(self.board)

    def __copy__(self) -> 'ChessState':
        return ChessState(self.board.copy())

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
        self.state = self.state.get_next_state(action, player)

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
