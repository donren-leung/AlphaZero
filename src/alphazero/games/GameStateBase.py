from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
import torch

type NumpyState = NDArray[np.int8]

class GameStateBase(ABC):
    __slots__ = ["state"]
    def __init__(self, state: Any):
        self.state = state

    @abstractmethod
    def get_legal_actions(self) -> NDArray[np.bool_]:
        ...

    @abstractmethod
    def get_next_state(self, action: int, player: int, copy: bool=True) -> Self:
        ...

    @abstractmethod
    def get_value_and_terminated(self, action: int) -> tuple[int, bool]:
        """
        Checks whether the game state (resulting after the queried action) is terminal.\
        If so, return the reward -1 to 1 for the player who made the queried action"""
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self) -> Self:
        ...

class GameStateNumpyBase(GameStateBase):
    __slots__ = ["state"]
    def __init__(self, state: NumpyState):
        self.state = state

    def to_tensor(self) -> torch.Tensor:
        # return torch.tensor(self.state, dtype=torch.int8).unsqueeze(0)
        return torch.from_numpy(
            np.stack([
                self.state == -1,
                self.state == 0,
                self.state == 1,
            ]).astype(np.int8, copy=False)
        )

    def neutral_state(self, player: int) -> Self:
        return self.__class__(self.state * player)
