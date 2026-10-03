from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray
import torch

type State = NDArray[np.int8]

class GameStateBase(ABC):
    __slots__ = ["state"]
    def __init__(self, state: State):
        self.state = state

    @abstractmethod
    def get_legal_actions(self, player: int) -> NDArray[np.bool_]:
        ...

    @abstractmethod
    def get_next_state(self, action: int, player: int, copy: bool=True) -> Self:
        ...

    # @abstractmethod
    # def check_win(self, action: int) -> bool:
    #     """
    #     Checks whether an action that's just been made (resulting in current
    #     game state) results in a win for the player who made that action.
    #     """
    #     ...

    @abstractmethod
    def get_value_and_terminated(self, action: int) -> tuple[int, bool]:
        """
        Checks whether the game state (resulting after the queried action) is terminal.\
        If so, return the reward -1 to 1 for the player who made the queried action"""
        ...

    def to_tensor(self) -> torch.Tensor:
        # return torch.tensor(self.state, dtype=torch.int8).unsqueeze(0)
        return torch.from_numpy(
            np.stack([
                self.state == -1,
                self.state == 0,
                self.state == 1,
            ]).astype(np.int8, copy=False)
        )

    # @classmethod
    # def neutral_state(cls, state: 'GameStateBase', player: int) -> 'GameStateBase':
    #     return cls(state.state * player)

    def neutral_state(self, player: int) -> Self:
        return self.__class__(self.state * player)

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self) -> Self:
        ...
