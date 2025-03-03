
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Type, final

from games.GameStateBase import GameStateBase

from numpy import int8, bool_
from numpy.typing import NDArray

GameStateT = TypeVar('GameStateT', bound='GameStateBase')

class GameBase(Generic[GameStateT], ABC):
    state_cls: Type[GameStateT]
    __slots__ = ["state", "waiting_for_result", "current_player"]

    def __init__(self, state: GameStateT | None = None, starting_player: int=1) -> None:
        if state is None:
            self.state = self.get_initial_state()
        else:
            self.state = state

        # For parallel games
        self.waiting_for_result = False
        self.current_player = starting_player

    @classmethod
    @abstractmethod
    def get_initial_state(cls) -> GameStateT:
        ...

    @abstractmethod
    def get_legal_actions(self) -> NDArray[bool_]:
        ...

    @property
    @abstractmethod
    def action_size(self) -> int:
        ...

    @final
    def make_move(self, action: int, player: int) -> None:
        self._make_move(action, player)
        self._switch_current_player()

    @abstractmethod
    def _make_move(self, action: int, player: int) -> None:
        ...

    # @abstractmethod
    # def check_win(self, action: int) -> bool:
    #     ...

    @abstractmethod
    def get_value_and_terminated(self, action: int):
        ...

    @final
    @classmethod
    def get_opponent(self, player: int) -> int:
        return player * -1

    def _switch_current_player(self) -> None:
        self.current_player = self.get_opponent(self.current_player)

    @abstractmethod
    def __repr__(self):
        ...

    def __str__(self) -> str:
        return self.state.__str__()
