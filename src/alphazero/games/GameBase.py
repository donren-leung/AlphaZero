
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Generic, TypeVar, ClassVar, Type, final, Literal

from .GameStateBase import ModelInput, GameStateBase, GameStateNumpyBase

import numpy as np
from numpy import int8, bool_
from numpy.typing import NDArray

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
GameStateNumpyT = TypeVar('GameStateNumpyT', bound='GameStateNumpyBase')

class GameBase(Generic[GameStateT], ABC):
    state_cls: Type[GameStateT]
    action_size: ClassVar[int]

    row_count: ClassVar[int]
    col_count: ClassVar[int]

    __slots__ = ["state",
                 "history",
                 "current_player"]

    def __init__(self, state: GameStateT | None = None, starting_player: int=1) -> None:
        if state is None:
            self.state = self.get_initial_state()
        else:
            self.state = state

        self.current_player: int = starting_player
        self.history: dict[str, list[Any]] = defaultdict(list)

    @classmethod
    @abstractmethod
    def get_initial_state(cls) -> GameStateT:
        ...

    @abstractmethod
    def get_legal_actions(self) -> NDArray[bool_]:
        ...

    def make_move(self, action: int) -> None:
        self._make_move(action, self.current_player)
        self._switch_current_player()

    @abstractmethod
    def _make_move(self, action: int, player: int) -> None:
        ...

    @abstractmethod
    def get_value_and_terminated(self, action: int) -> tuple[Literal[1], Literal[True]] | \
                                                        tuple[Literal[0], Literal[True]] | \
                                                        tuple[Literal[0], Literal[False]]:
        ...

    @final
    @classmethod
    def get_opponent(cls, player: int) -> int:
        return player * -1

    def _switch_current_player(self) -> None:
        self.current_player = self.get_opponent(self.current_player)

    @abstractmethod
    def encode_for_model(self) -> ModelInput:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        return self.state.__str__()

class GameBaseNumpy(Generic[GameStateNumpyT], GameBase[GameStateNumpyT]):
    __slots__ = ["state",
                 "history",
                 "current_player"]

    def __init__(self, state: GameStateNumpyT | None = None, starting_player: int=1) -> None:
       super().__init__(state, starting_player)

    def encode_for_model(self) -> ModelInput:
        state = self.state.neutral_state(self.current_player)
        return np.stack([
            state.state == -1,
            state.state == 0,
            state.state == 1,
        ]).astype(np.uint8, copy=False)

