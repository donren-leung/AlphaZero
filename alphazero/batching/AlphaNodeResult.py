from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np

from games.GameStateBase import GameStateBase

#  value scalar, policy vector, is_terminal
SimulationReturnType = tuple[int, NDArray[np.float32], bool]

@dataclass(slots=True, frozen=True)
class NodeRequest:
    worker_id: int
    thread_id: int

    curr_player: int

    action_and_state: tuple[int, GameStateBase]

@dataclass(slots=True, frozen=True)
class NodeResponse:
    worker_id: int
    thread_id: int
    result:    SimulationReturnType
