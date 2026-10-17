from typing import ClassVar

import chess
import numpy as np
from numpy.typing import NDArray

from .GameStateBase import GameStateBase
from .GameBase import GameBase
from .ChessUtils import encode_move, decode_action, CHESS_ACTION_SIZE

type State = chess.Board

class ChessState(GameStateBase):
    __slots__ = ["row_count", "col_count", "in_a_row"]
    def __init__(self, state: State):
        self.state = state
        self.row_count = 8
        self.col_count = 8

    def get_legal_actions(self) -> NDArray[np.bool_]:
        player = 1 if self.state.turn == chess.WHITE else -1
        legal_actions = np.zeros(CHESS_ACTION_SIZE, dtype=np.bool_)
        for move in self.state.legal_moves:
            action = encode_move(self.state, move, player)
            legal_actions[action] = True
        return legal_actions

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'ChessState':
        if copy:
            new_state = self.state.copy()
        else:
            new_state = self.state

        move = decode_action(new_state, action, player)
        new_state.push(move)
        return ChessState(new_state)

    def get_value_and_terminated(self, action: int=0):
        outcome = self.state.outcome(claim_draw=True)
        if outcome is None:
            return 0, False
        if outcome.winner is None:
            return 0, True
        # Checkmate. The player who just moved is the winner, so the current player is the loser.
        assert outcome.winner is not self.state.turn
        return 1, True

    def __str__(self) -> str:
        return str(self.state)

    def __copy__(self) -> 'ChessState':
        return ChessState(self.state.copy())

class ChessGame(GameBase[ChessState]):
    def __init__(self, state: ChessState | None=None) -> None:
        super().__init__(state)

    @classmethod
    def get_initial_state(cls) -> ChessState:
        return ChessState(chess.Board())

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return self.state.get_legal_actions()

    def _make_move(self, action: int, player: int) -> None:
        self.state = self.state.get_next_state(action, player)

    def get_value_and_terminated(self, action: int):
        return self.state.get_value_and_terminated(action)

    def __repr__(self) -> str:
        return "Chess"

    def __str__(self) -> str:
        return self.state.__str__()
