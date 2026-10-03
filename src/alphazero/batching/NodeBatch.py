from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from alphazero.games.GameStateBase import GameStateBase

# visits, total value, terminal
SimulationReturnType = tuple[int, float, bool]

@dataclass(slots=True, frozen=True)
class NodeBatchRequest:
    worker_id: int
    thread_id: int

    curr_player: int
    target_sims: int

    states_and_actions: list[tuple[int, GameStateBase]]

@dataclass(slots=True, frozen=True)
class NodeBatchResponse:
    worker_id: int
    thread_id: int
    results:   list[SimulationReturnType]

    def to_tuple(self) -> tuple[int, float, bool]:
        assert len(self.results) == 1
        return (self.results[0][0], self.results[0][1], self.results[0][2])

# policy, value, terminal
AZ_SimulationReturnType = tuple[npt.NDArray[np.float32], float]

@dataclass(slots=True, frozen=True)
class AZ_NodeBatchRequest:
    worker_id: int
    thread_id: int

    curr_player: int
    state: GameStateBase

@dataclass(slots=True, frozen=True)
class AZ_NodeBatchResponse:
    worker_id:  int
    # keyed by thread_id
    results:    list[tuple[int, AZ_SimulationReturnType]]
