import logging
import math
import time

from multiprocessing.context import BaseContext
from queue import Queue, Empty
from threading import Thread, Lock, Condition
from typing import Generic, TypeVar, Type


from .Pool import mpQueueGen
from .NodeBatch import NodeBatchRequest, NodeBatchResponse, SimulationReturnType
from alphazero.games.GameBase import GameBase, GameStateBase
from alphazero.MCTS.MCTS_batch import MCTS_Factory, Node

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
GameT = GameBase[GameStateT]

class GameWorker(Generic[GameStateT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position, for each parallel game of specified game type.
    """
    metrics_interval_s = 0.5

    def __init__(self, game_type: type[GameBase[GameStateT]], *,
                 num_games: int, output_games: mpQueueGen[tuple[str, GameBase[GameStateT]]],
                 in_queue: mpQueueGen[NodeBatchResponse],
                 out_queue: mpQueueGen[NodeBatchRequest],
                 metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]],
                 worker_id: int, MCTS_factory: MCTS_Factory):
        self.game_type = game_type
        self.num_games = num_games

        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[NodeBatchResponse]                  = in_queue
        self.metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]]= metrics_queue

        self.out_queue:     mpQueueGen[NodeBatchRequest]                   = out_queue
        self.output_games:  mpQueueGen[tuple[str, GameBase[GameStateT]]]   = output_games

        # Consider below zipped by thread index
        self.thread_inbox:  list[NodeBatchResponse | None]  = []
        self.threads:       list[Thread]                    = []
        self.inbox_cv:      list[Condition]                 = []

        # Metrics
        self.total_moves = 0
        self.sent_batches = 0
        self.sent_nodes = 0
        self.received_batches = 0
        self.received_nodes = 0
        self.active_games = 0

    def run(self) -> None:
        self.metrics_lock = Lock()
        self.outbound_cv = Condition()

        self.staging_out_queue: Queue[NodeBatchRequest] = Queue()
        Thread(target=self.inbound_mailman_d, args=[], daemon=True).start()
        Thread(target=self.metrics_d, args=[], daemon=True).start()

        for i in range(self.num_games):
            new_game_instance = self.game_type()
            new_inbox_cv = Condition()

            t = Thread(target=self.run_game, args=[i, new_game_instance])
            t.name = f"w{self.worker_id}.g{i}"

            self.thread_inbox.append(None)
            self.threads.append(t)
            self.inbox_cv.append(new_inbox_cv)
            self.active_games += 1

        for thread in self.threads:
            thread.start()

    def inbound_mailman_d(self) -> None:
        """
        'Mailman' thread -- Gets BatchResponses from the MP world and
        in the local process puts it into the correct per-thread mailbox.
        """
        while True:
            # Blocking call to get a BatchResponse from the MP queue.
            response = self.in_queue.get()
            assert response.worker_id == self.worker_id
            assert response.thread_id >= 0 and response.thread_id < len(self.threads)
            assert self.thread_inbox[response.thread_id] is None

            with self.metrics_lock:
                self.received_batches += 1
                self.received_nodes += len(response.results)

            # Open (with mutex) the right thread's mailbox, insert the
            # received BatchResponse and notify the waiting thread.
            self.inbox_cv[response.thread_id].acquire()
            self.thread_inbox[response.thread_id] = response
            self.inbox_cv[response.thread_id].notify()
            self.inbox_cv[response.thread_id].release()


    def metrics_d(self) -> None:
        while True:
            time.sleep(self.metrics_interval_s)
            self.metrics_queue.put((
                self.worker_id,
                self.total_moves,
                self.sent_batches,
                self.sent_nodes,
                self.received_batches,
                self.received_nodes
            ))

    def run_game(self, thread_id: int, game: GameT) -> None:
        logging.info(f"thread {thread_id} started")

        # player
        first_player = game.current_player
        while True:
            logging.debug(f"thread {thread_id} iteration {len(game.action_history)}: {game.action_history}")

            # Game after last move is still going, create new MCTS instance.
            MCTS_instance = self.MCTS_factory.make_instance(game=game)

            while MCTS_instance.root.visits < MCTS_instance.rollouts:
                # While not enough rollouts:
                logging.debug(f"thread {thread_id} {MCTS_instance.root.visits} visits, not enough")
                # Make next batch
                logging.debug(f"{MCTS_instance.root=}")
                node_or_nodes, request = MCTS_instance.one_round_batch(self.worker_id, thread_id)

                if isinstance(request, NodeBatchRequest):
                    # Send batch request to worker pool.
                    assert isinstance(node_or_nodes, list), node_or_nodes
                    self.out_queue.put(request)
                    with self.metrics_lock:
                        self.sent_batches += 1
                        self.sent_nodes += len(request.states_and_actions)

                    # Get or wait (block thread) for the result.
                    self.inbox_cv[thread_id].acquire()
                    # NOTE: Is this while loop strictly needed??
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
                    # Cached result; no request is sent.
                    logging.debug(f"thread {thread_id} (cached) batchresponse")
                    assert(isinstance(node_or_nodes, Node))
                    with self.metrics_lock:
                        self.received_batches += 1
                        self.received_nodes += 1

                    node_or_nodes.backpropogate(*request.to_tuple())

                MCTS_instance.root.print_children(0, limit=2)

            # Make move
            root = MCTS_instance.root
            children_details = [(
                    child.visits / root.visits if MCTS_instance.rollouts else 0,
                    # (child.value_sum / child.visits) if child.visits else 0,
                    # root.get_ucb(child),
                    # child.visits,
                    child.parent_action if child.parent_action is not None else -1)
                for child in root.children
            ]
            best_action = max(children_details, key=lambda x: x[0])[1]
            game.make_move(best_action)
            logging.debug(f"thread {thread_id} made move {best_action}")
            with self.metrics_lock:
                self.total_moves += 1

            value, terminated = game.get_value_and_terminated(best_action)
            if terminated:
                assert len(game.state_history) == len(game.action_prob_history)
                hist_player = first_player
                last_player = game.get_opponent(game.current_player)

                for _ in game.action_prob_history:
                    outcome = value if hist_player == last_player else -value
                    game.outcome.append(outcome)
                    hist_player = game.get_opponent(hist_player)

                break

        # Game finished
        logging.info(f"thread {thread_id} finished game")
        self.output_games.put((f"{self.worker_id}_{thread_id}", game))

        with self.metrics_lock:
            self.active_games -= 1
