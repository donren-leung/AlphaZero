from abc import ABC, abstractmethod

from numpy import int8, bool_
from numpy.typing import NDArray

type State = NDArray[int8]

class GameStateBase(ABC):
    __slots__ = ["state"]
    def __init__(self, state: State):
        self.state = state

    @abstractmethod
    def get_legal_actions(self, player: int) -> NDArray[bool_]:
        ...

    @abstractmethod
    def get_next_state(self, action: int, player: int, copy: bool=True) -> 'GameStateBase':
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

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def __copy__(self) -> 'GameStateBase':
        ...
