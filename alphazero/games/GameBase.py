
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type

from games.GameStateBase import GameStateBase

from numpy import int8, bool_
from numpy.typing import NDArray

GameStateT = TypeVar('GameStateT', bound='GameStateBase')

class GameBase(Generic[GameStateT], ABC):
    state_cls: Type[GameStateT]

    def __init__(self, state: GameStateT | None = None) -> None:
        if state is None:
            self.state = self.get_initial_state()
        else:
            self.state = state

    @classmethod
    @abstractmethod
    def get_initial_state(cls) -> GameStateT:
        ...

    @abstractmethod
    def get_legal_actions(self) -> NDArray[bool_]:
        ...

    @abstractmethod
    def make_move(self, action: int, player: int) -> None:
        ...

    @abstractmethod
    def check_win(self, action: int) -> bool:
        ...

    @abstractmethod
    def get_value_and_terminated(self, action: int):
        ...

    @abstractmethod
    def get_opponent(self, player: int) -> int:
        ...

    @abstractmethod
    def __repr__(self):
        ...

    def __str__(self) -> str:
        return self.state.__str__()
