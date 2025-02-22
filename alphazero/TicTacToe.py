import numpy as np
from numpy.typing import NDArray

type State = NDArray[np.int8]

class TicTacToeState:
    def __init__(self, state: State):
        self.state = state
        self.row_count = state.shape[0]
        self.col_count = state.shape[1]

    def get_valid_moves(self) -> NDArray[np.bool_]:
        return (self.state.reshape(-1) == 0).astype(np.bool_)
    
    def get_next_state(self, action: int, player: int) -> 'TicTacToeState':
        row = action // self.col_count
        col = action % self.col_count

        new_state = np.copy(self.state)
        new_state[row, col] = player
        return TicTacToeState(new_state)
    
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
    
    def __str__(self) -> str:
        return str(self.state)

class TicTacToeGame:
    def __init__(self):
        self.row_count: int = 3
        self.col_count: int = 3
        self.action_size: int = self.row_count * self.col_count
        self.state: TicTacToeState = self.get_initial_state()

    def get_initial_state(self) -> TicTacToeState:
        return TicTacToeState(np.zeros((self.row_count, self.col_count), dtype=np.int8))

    def get_valid_moves(self) -> NDArray[np.bool_]:
        return self.state.get_valid_moves()
    
    def make_move(self, action: int, player: int) -> None:
        self.state = self.state.get_next_state(action, player)
      
    def check_win(self, action: int) -> bool:
        return self.state.check_win(action)
    
    def get_value_and_terminated(self, action: int):
        if self.state.check_win(action):
            return 1, True
        elif not np.any(self.state.get_valid_moves()):
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player

    def __str__(self) -> str:
        return self.state.__str__()
