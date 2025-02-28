from games.GameStateBase import GameStateBase
from games.GameBase import GameBase

import numpy as np
from numpy.typing import NDArray

type State = NDArray[np.int8]

class TicTacToeState(GameStateBase):
    __slots__ = ["row_count", "col_count"]
    def __init__(self, state: State):
        self.state = state
        self.row_count = state.shape[0]
        self.col_count = state.shape[1]

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return (self.state.reshape(-1) == 0).astype(np.bool_)

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'TicTacToeState':
        assert self.get_legal_actions()[action], f"action {action} not valid"
        row = action // self.col_count
        col = action % self.col_count

        if copy:
            new_state = np.copy(self.state)
            new_state[row, col] = player
            return TicTacToeState(new_state)
        else:
            self.state[row, col] = player
            return self

    # Checks if a move that's just been made results in a win for that player
    def check_win(self, action: int) -> bool:
        state = self.state
        row = action // self.col_count
        col = action % self.col_count
        player = state[row, col]

        return (
            np.sum(state[row, :]) == player * self.col_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.fliplr(state))) == player * self.row_count
        )

    def get_value_and_terminated(self, action: int):
        if self.check_win(action):
            return 1, True
        elif not np.any(self.get_legal_actions()):
            return 0, True
        return 0, False

    def __str__(self) -> str:
        return str(self.state)

    def __copy__(self) -> 'TicTacToeState':
        return TicTacToeState(np.copy(self.state))

class TicTacToeGame(GameBase[TicTacToeState]):
    row_count: int = 3
    col_count: int = 3
    action_size: int = row_count * col_count

    def __init__(self, state: TicTacToeState | None = None) -> None:
        super().__init__(state)

    @classmethod
    def get_initial_state(cls) -> TicTacToeState:
        return TicTacToeState(np.zeros((cls.row_count, cls.col_count), dtype=np.int8))

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return self.state.get_legal_actions()

    def make_move(self, action: int, player: int) -> None:
        self.state = self.state.get_next_state(action, player)

    def check_win(self, action: int) -> bool:
        return self.state.check_win(action)

    def get_value_and_terminated(self, action: int):
        return self.state.get_value_and_terminated(action)

    def get_opponent(self, player: int) -> int:
        return -player

    def __repr__(self):
        return "TicTacToe"

    def __str__(self) -> str:
        return self.state.__str__()
