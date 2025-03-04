import logging
import sys

from math import nan
from multiprocessing import Queue as mpQueue
from queue import Queue
from threading import Thread, local, Condition
from typing import Generic, TypeVar, Type

from batching.NodeBatch import NodeBatchRequest, NodeBatchResponse
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from games.GameBase import GameBase
from games.GameStateBase import GameStateBase
from games.TicTacToe import TicTacToeGame
from games.ConnectFour import ConnectFourGame
from MCTS_batch import MCTS_Factory, Node, simulate_

GameT = TypeVar('GameT', bound='GameBase')

T = TypeVar("T")

class mpQueueGen(Generic[T]):
    def __init__(self, *args, **kwargs):
        self._queue = mpQueue(*args, **kwargs)

    def put(self, item: T) -> None:
        self._queue.put(item)

    def get(self) -> T:
        return self._queue.get()

    def empty(self):
        return self._queue.empty()

class GameWorker(Generic[GameT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position for each parallel game of specified game type.
    """
    def __init__(self, game_type: type[GameT], *,
                 num_games: int, output_games: mpQueueGen[tuple[str, GameT]],
                 in_queue: mpQueueGen[NodeBatchResponse], out_queue: mpQueueGen[NodeBatchRequest],
                 worker_id: int, MCTS_factory: MCTS_Factory):
        self.game_type = game_type
        self.num_games = num_games
        # self.finished_game_list: Queue[GameT] = Queue()
        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[NodeBatchResponse]   = in_queue
        self.out_queue:     mpQueueGen[NodeBatchRequest]    = out_queue
        self.output_games:  mpQueueGen[tuple[str, GameT]]   = output_games
        # Consider below zipped by thread index
        self.thread_inbox:  list[NodeBatchResponse | None]  = []
        self.threads:       list[Thread]                    = []
        self.inbox_cv:      list[Condition]                 = []

    def run(self) -> None:
        for i in range(self.num_games):
            new_game_instance = self.game_type()
            new_inbox_cv = Condition()

            t = Thread(target=self.run_game, args=[i, new_game_instance])
            t.name = f"w{self.worker_id}.g{i}"

            self.thread_inbox.append(None)
            self.threads.append(t)
            self.inbox_cv.append(new_inbox_cv)

        Thread(target=self.daemon_thread, args=[], daemon=True).start()

        for i, thread in enumerate(self.threads):
            thread.start()

    def daemon_thread(self) -> None:
        while True:
            response = self.in_queue.get()
            assert response.worker_id == self.worker_id
            assert response.thread_id >= 0 and response.thread_id < len(self.threads)
            assert self.thread_inbox[response.thread_id] is None

            self.inbox_cv[response.thread_id].acquire()
            self.thread_inbox[response.thread_id] = response
            self.inbox_cv[response.thread_id].notify()
            self.inbox_cv[response.thread_id].release()

    def run_game(self, thread_id: int, game: GameT) -> None:
        logging.info(f"thread {thread_id} started")
        while True:
            # Check if game has ended (which can't happen on first move)
            if len(game.action_history) == 0:
                pass
            else:
                value, terminated = game.get_value_and_terminated(game.action_history[-1])
                if terminated:
                    break

            logging.debug(f"thread {thread_id} iteration {len(game.action_history)}: {game.action_history}")
            # Game after last move is still going, create new MCTS
            MCTS_instance = self.MCTS_factory.make_instance(game=game)

            while MCTS_instance.root.visits < MCTS_instance.rollouts:
                # While not enough rollouts:
                logging.debug(f"thread {thread_id} {MCTS_instance.root.visits} visits, not enough")
                # Make next batch
                logging.debug(f"{MCTS_instance.root=}")
                node_or_nodes, request = MCTS_instance.one_round_batch(self.worker_id, thread_id)

                if isinstance(request, NodeBatchRequest):
                    # Send batch
                    assert isinstance(node_or_nodes, list), node_or_nodes
                    self.out_queue.put(request)

                    # Get the result
                    self.inbox_cv[thread_id].acquire()
                    while self.thread_inbox[thread_id] is None:
                        self.inbox_cv[thread_id].wait()
                    response = self.thread_inbox[thread_id]
                    self.thread_inbox[thread_id] = None
                    self.inbox_cv[thread_id].release()
                    assert response is not None

                    # Backpropogate
                    visits_and_values: list[tuple[int, float]] = []
                    parent = node_or_nodes[0].parent

                    for node, result in zip(node_or_nodes, response.results):
                        visits_and_values.append(result[:2])
                        node.backpropogate(*result, stop_at_node=node)

                    if parent is not None:
                        total_visits = sum(visits for visits, _ in visits_and_values)
                        total_values = sum(-values for _, values in visits_and_values)
                        parent.backpropogate(total_visits, total_values, False)

                else:
                    # Cached
                    logging.debug(f"thread {thread_id} (cached) batchresponse")
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
            logging.debug(f"thread {thread_id} made move {best_action}")

        # Game finished
        logging.info(f"thread {thread_id} finished game")
        self.output_games.put((f"{self.worker_id}_{thread_id}", game))
