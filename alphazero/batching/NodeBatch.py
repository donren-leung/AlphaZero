from dataclasses import dataclass

from games.GameStateBase import GameStateBase

# visits, total value, terminal
SimulationReturnType = tuple[int, float, bool]

@dataclass(slots=True, frozen=True)
class NodeBatchRequest:
    worker_id: int
    thread_id: int

    curr_player: int
    target_sims: int

    action_and_state: list[tuple[int, GameStateBase]]
    # node_hash: list[int]

@dataclass(slots=True, frozen=True)
class NodeBatchResponse:
    worker_id: int
    thread_id: int
    results:   list[SimulationReturnType]

    def to_tuple(self) -> tuple[int, float, bool]:
        assert len(self.results) == 1
        return (self.results[0][0], self.results[0][1], self.results[0][2])
