import logging
import sys

from math import nan
from queue import Queue
from threading import Thread, local, Condition
from typing import Generic, TypeVar, Type

from batching.NodeBatch import NodeBatchRequest, NodeBatchResponse, SimulationReturnType
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from games.GameBase import GameBase
from games.GameStateBase import GameStateBase
from games.TicTacToe import TicTacToeGame
from games.ConnectFour import ConnectFourGame
from MCTS_batch import MCTS_Factory, Node, simulate_

GameT = TypeVar('GameT', bound='GameBase')

class GameWorker(Generic[GameT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position for each parallel game of specified game type.
    """
    def __init__(self, game_type: type[GameT], num_games: int,
                 responses: Queue[NodeBatchResponse], *,
                 worker_id: int, MCTS_factory: MCTS_Factory):
        self.game_type = game_type
        self.num_games = num_games
        self.finished_game_list: Queue[GameT] = Queue()
        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory

        # Threading stuff
        self.responses:     Queue[NodeBatchResponse]    = responses
        # Consider below zipped by thread index
        self.thread_inbox:  list[NodeBatchResponse]     = []
        self.threads:       list[Thread]                = []
        self.inbox_cv:      list[Condition]             = []
        self.executor = ProcessPoolExecutor(max_workers=MCTS_factory.processes)

    def spawn_games(self) -> None:
        for i in range(self.num_games):
            new_game_instance = self.game_type()
            new_inbox_cv = Condition()

            t = Thread(target=self.run_game, args=[i, new_game_instance,
                                                   new_inbox_cv, self.executor])
            t.name = f"w{self.worker_id}.g{i}"

            # self.thread_inbox.append()
            self.threads.append(t)
            self.inbox_cv.append(new_inbox_cv)

        for i, thread in enumerate(self.threads):
            thread.start()

    def collect_games(self) -> list[GameT]:
        for thread in self.threads:
            thread.join()

        return list(self.finished_game_list.queue)

    def run_game(self, id_: int, game: GameT, inbox_cv: Condition, executor: ProcessPoolExecutor) -> None:
        logging.debug(f"worker {id_} started")
        while True:
            if len(game.action_history) == 0:
                pass
            else:
                value, terminated = game.get_value_and_terminated(game.action_history[-1])
                if terminated:
                    break

            logging.debug(f"worker {id_} iteration {len(game.action_history)}: {game.action_history}")
            # Game after last move is still going, create new MCTS
            MCTS_instance = self.MCTS_factory.make_instance(game=game)

            while MCTS_instance.root.visits < MCTS_instance.rollouts:
                # While not enough rollouts:
                logging.debug(f"worker {id_} {MCTS_instance.root.visits} visits, not enough")
                # Make next batch
                logging.debug(MCTS_instance.root)

                # Send batch

                # Get the result
                node_or_nodes, request = MCTS_instance.one_round_batch()

                if isinstance(request, NodeBatchRequest):
                    logging.debug(f"worker {id_} batchrequest")
                    assert isinstance(node_or_nodes, list), node_or_nodes
                    pending_simulations: dict[Future[SimulationReturnType], Node] = {}
                    for node, (action, state) in zip(node_or_nodes, request.action_and_state):
                        # logging.debug(f"{node}, {node.parent}, {node.parent_action}")
                        future = executor.submit(simulate_,
                                                    state,
                                                    request.curr_player,
                                                    action,
                                                    target_sims=request.target_sims,
                                                    debug=MCTS_Factory.debug
                                                    )
                        pending_simulations[future] = node

                    future_iterator = as_completed(pending_simulations.keys())

                    while True:
                        try:
                            future = next(future_iterator)
                            sim_result = future.result()
                            node = pending_simulations.pop(future)

                            node.backpropogate(*sim_result)
                            logging.debug(f"worker {id_} backpropping {sim_result} to {node.parent_action}")
                        except StopIteration:
                            logging.debug(f"worker {id_} stopiteration")
                            break

                else:
                    # Response
                    logging.debug(f"worker {id_} (cached) batchresponse")
                    assert(isinstance(node_or_nodes, Node))
                    node_or_nodes.backpropogate(*request.to_tuple())

                MCTS_instance.root.print_children(0, limit=2)

            # Make move
            root = MCTS_instance.root
            children_details = [(
                    child.visits / root.visits if MCTS_instance.rollouts else nan,
                    (child.value_sum / child.visits) if child.visits else nan,
                    root.get_ucb(child),
                    child.visits,
                    child.parent_action if child.parent_action is not None else -1)
                for child in root.children
            ]
            best_action = max(children_details, key=lambda x: x[0])[4]
            game.make_move(best_action)
            logging.debug(f"worker {id_} made move {best_action}")

        # Game finished
        logging.debug(f"worker {id_} finished game")
        self.finished_game_list.put(game)


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    game_worker = GameWorker(ConnectFourGame, 4, Queue(),
                             worker_id=0, MCTS_factory=MCTS_Factory(10000, 20, 15))

    game_worker.spawn_games()
    games = game_worker.collect_games()

    for game in games:
        print(game.action_history)
        print(game.state)
