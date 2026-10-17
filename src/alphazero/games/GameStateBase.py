from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch

type ModelInput = NDArray[np.uint8]
type NumpyState = NDArray[np.int8]

class CompactRepr(Protocol):
    def unpack(self) -> ModelInput:
        ...

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

    # @abstractmethod
    # def encode_for_model(self, player: int) -> NDArray[np.uint8]:
    #     ...

    # @abstractmethod
    # def neutral_state(self, player: int) -> Self:
    #     ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self) -> Self:
        ...

    @abstractmethod
    def to_compact_repr(self, player: int) -> CompactRepr:
        ...

    # Something to which you can call _unpack and get back a model-ready representation of the game state.

class GameStateNumpyBase(GameStateBase):
    __slots__ = ["state"]
    def __init__(self, state: NumpyState):
        self.state = state

    def neutral_state(self, player: int) -> Self:
        return self.__class__(self.state * player)

    def to_compact_repr(self, player: int) -> CompactRepr:
        state = self.neutral_state(player)
        return CompactNumpyState(state.state)

@dataclass(frozen=True, slots=True)
class CompactNumpyState:
    state: NumpyState

    def unpack(self) -> ModelInput:
        return np.stack([
            self.state == -1,
            self.state == 0,
            self.state == 1,
        ]).astype(np.uint8, copy=False)
