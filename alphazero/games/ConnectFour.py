from games.GameStateBase import GameStateBase
from games.GameBase import GameBase

import numpy as np
from numpy.typing import NDArray

type State = NDArray[np.int8]

class ConnectFourState(GameStateBase):
    __slots__ = ["row_count", "col_count", "in_a_row"]
    def __init__(self, state: State):
        self.state = state
        self.row_count = state.shape[0]
        self.col_count = state.shape[1]
        self.in_a_row = 4

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return (self.state[0] == 0).astype(np.bool_)

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'ConnectFourState':
        row = np.max(np.where(self.state[:, action] == 0))

        if copy:
            new_state = np.copy(self.state)
            new_state[row, action] = player
            return ConnectFourState(new_state)
        else:
            self.state[row, action] = player
            return self

    # Checks if a move that's just been made results in a win for that player
    def check_win(self, action: int) -> bool:
        if action == None:
            return False

        non_zero_indices = np.where(self.state[:, action] != 0)[0]
        if non_zero_indices.size == 0:
            # No pieces in this column, so no win is possible here.
            return False

        row = np.min(np.where(self.state[:, action] != 0))
        column = action
        player = self.state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.col_count
                    or self.state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )

    def get_value_and_terminated(self, action: int):
        if self.check_win(action):
            return 1, True
        elif not np.any(self.get_legal_actions()):
            return 0, True
        return 0, False

    def __str__(self) -> str:
        return str(self.state)

    def __copy__(self) -> 'ConnectFourState':
        return ConnectFourState(np.copy(self.state))

class ConnectFourGame(GameBase[ConnectFourState]):
    row_count: int = 6
    column_count: int = 7
    action_size: int = column_count
    in_a_row: int = 4

    def __init__(self, state: ConnectFourState | None=None) -> None:
        super().__init__(state)

    @classmethod
    def get_initial_state(cls) -> ConnectFourState:
        return ConnectFourState(np.zeros((cls.row_count, cls.column_count), dtype=np.int8))

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

    def __repr__(self) -> str:
        return "ConnectFour"

    def __str__(self) -> str:
        return self.state.__str__()
