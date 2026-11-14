from typing import ClassVar, Literal

from .GameStateBase import GameStateNumpyBase
from .GameBase import GameBaseNumpy

import numpy as np
from numpy.typing import NDArray

type State = NDArray[np.int8]

class TicTacToeState(GameStateNumpyBase):
    __slots__ = ["row_count", "col_count"]
    def __init__(self, state: State, action_history: list[int]=[], turn: Literal[1, -1] = 1):
        super().__init__(state, action_history.copy(), turn)
        self.row_count = state.shape[0]
        self.col_count = state.shape[1]

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return (self.board.reshape(-1) == 0).astype(np.bool_)

    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'TicTacToeState':
        assert self.get_legal_actions()[action], f"action {action} not valid"
        row = action // self.col_count
        col = action % self.col_count

        self.switch_player()
        if copy:
            new_state = np.copy(self.board)
            new_state[row, col] = player
            return TicTacToeState(new_state, self.action_history + [action], self.turn)
        else:
            self.board[row, col] = player
            self.action_history.append(action)
            return self

    def push_to_board(self, move: int) -> None:
        self.get_next_state(move, self.turn, copy=False)

    def pop_board(self) -> int:
        last_action = self.action_history.pop()
        row = last_action // self.col_count
        col = last_action % self.col_count
        assert self.board[row, col] != 0, "Trying to pop an empty square"

        self.board[row, col] = 0
        self.switch_player()
        return last_action

    def _check_winner(self, action: int) -> int:
        """
        Checks if a move that's just been made results in a win for that player
        """
        state = self.board
        row, col = divmod(action, self.col_count)
        player = state[row, col]

        # Actually given an empty square
        if player == 0:
            return 0

        return player if (
            np.sum(state[row, :]) == player * self.col_count
            or np.sum(state[:, col]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.fliplr(state))) == player * self.row_count
        ) else 0

    def _check_winner_full(self) -> int:
        """
        Check all rows, columns, and diagonals for either player's win.
        Return the winning player (1 or -1), or 0 if no winner.
        """
        state = self.board

        for player in (1, -1):
            matches = state == player

            if (
                np.any(np.all(matches, axis=0))  # Columns
                or np.any(np.all(matches, axis=1))  # Rows
                or np.all(np.diag(matches))  # Main diagonal
                or np.all(np.diag(np.fliplr(matches)))  # Anti-diagonal
            ):
                return player

        return 0

    def get_value_and_terminated(self):
        if len(self.action_history) == 0:
            winner = self._check_winner_full()
        else:
            winner = (
                int(self.board.flat[self.action_history[-1]])
                if self._check_winner(self.action_history[-1])
                else 0
            )

        if winner != 0:
            # The player who just moved is the winner, so the current player is the loser.
            assert winner != self.turn
            return 1, True
        elif not np.any(self.board == 0):
            return 0, True
        else:
            return 0, False

    def __str__(self) -> str:
        return str(self.board)

    def __copy__(self) -> 'TicTacToeState':
        return TicTacToeState(np.copy(self.board))

class TicTacToeGame(GameBaseNumpy[TicTacToeState]):
    row_count: ClassVar[int] = 3
    col_count: ClassVar[int] = 3
    action_size: ClassVar[int] = row_count * col_count

    def __init__(self, state: TicTacToeState | None = None) -> None:
        super().__init__(state)

    @classmethod
    def get_initial_state(cls) -> TicTacToeState:
        return TicTacToeState(np.zeros((cls.row_count, cls.col_count), dtype=np.int8))

    def get_legal_actions(self) -> NDArray[np.bool_]:
        return self.state.get_legal_actions()

    def _make_move(self, action: int, player: int) -> None:
        self.state = self.state.get_next_state(action, player)

    def get_value_and_terminated(self):
        return self.state.get_value_and_terminated()

    def __repr__(self):
        return "TicTacToe"

    def __str__(self) -> str:
        return self.state.__str__()
