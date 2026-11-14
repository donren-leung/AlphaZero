from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

type ModelInput = NDArray[np.uint8]
type NumpyState = NDArray[np.int8]
MoveT = TypeVar("MoveT")

class CompactRepr(Protocol):
    def unpack(self) -> ModelInput:
        ...

class GameStateBase(Generic[MoveT], ABC):
    __slots__ = ["board"]
    def __init__(self, board: Any):
        self.board = board

    @abstractmethod
    def get_legal_actions(self) -> NDArray[np.bool_]:
        ...

    @abstractmethod
    def get_next_state(self, action: int, player: int, copy: bool=True) -> Self:
        ...

    @abstractmethod
    def encode_move(self, move: MoveT) -> int:
        ...

    @abstractmethod
    def decode_action(self, action: int, player: int) -> MoveT:
        ...

    @abstractmethod
    def push_to_board(self, move: MoveT) -> None:
        ...

    @abstractmethod
    def pop_board(self) -> MoveT:
        ...

    @abstractmethod
    def move_repr(self, move: MoveT) -> str:
        ...

    @abstractmethod
    def get_value_and_terminated(self) -> tuple[int, bool]:
        """
        Checks whether the game state (resulting after the queried action) is terminal.\
        If so, return the reward -1 to 1 for the player who made the queried action"""
        ...

    @abstractmethod
    def encode_for_model(self, player: int) -> NDArray[np.uint8]:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self) -> Self:
        ...

    @abstractmethod
    def to_compact_repr(self, player: int) -> CompactRepr:
        """
        Something to which you can call _unpack and get back a model-ready representation of the game state.
        """
        ...

class GameStateNumpyBase(GameStateBase[int]):
    __slots__ = ["board", "action_history", "turn"]
    def __init__(self, board: NumpyState, action_history: list[int], turn: Literal[1, -1]):
        self.board = board
        self.action_history = action_history
        self.turn: Literal[1, -1] = turn

    def neutral_state(self, player: int) -> Self:
        return self.__class__(self.board * player, self.action_history.copy(), self.turn)

    def switch_player(self) -> None:
        self.turn *= -1 # type: ignore

    def encode_move(self, move: int) -> int:
        return move

    def decode_action(self, action: int, player: int) -> int:
        return action

    def move_repr(self, move: int) -> str:
        return str(move)

    def to_compact_repr(self, player: int) -> CompactRepr:
        state = self.neutral_state(player)
        return CompactNumpyState(state.board)

    def encode_for_model(self, player: int) -> ModelInput:
        state = self.neutral_state(player)
        return np.stack([
            state.board == -1,
            state.board == 0,
            state.board == 1,
        ]).astype(np.uint8, copy=False)

@dataclass(frozen=True, slots=True)
class CompactNumpyState:
    state: NumpyState

    def unpack(self) -> ModelInput:
        return np.stack([
            self.state == -1,
            self.state == 0,
            self.state == 1,
        ]).astype(np.uint8, copy=False)
