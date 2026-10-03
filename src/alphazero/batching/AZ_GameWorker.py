import logging
import time

from queue import Queue, Empty
from threading import Thread, Lock, Condition
from typing import Generic, TypeVar, Type

import numpy as np
import numpy.typing as npt

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, AZ_SimulationReturnType
from .Pool import mpQueueGen
from alphazero.games.GameBase import GameBase, GameStateBase
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory, Node

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
GameT = GameBase[GameStateT]

class GameWorker(Generic[GameStateT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position, for each parallel game of specified game type.
    """
    metrics_interval_s = 0.5
    MAX_WAIT_S = 0.5
    RECHECK_INTERVAL_S = 0.1
    WAIT_FRAC = 0.4

    def __init__(self, game_type: type[GameBase[GameStateT]], *,
                 num_games: int, output_games: mpQueueGen[tuple[str, GameBase[GameStateT]]],
                 in_queue: mpQueueGen[AZ_NodeBatchResponse],
                 out_queue: mpQueueGen[list[AZ_NodeBatchRequest] | None],
                 metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]],
                 worker_id: int, MCTS_factory: MCTS_Factory):
        self.game_type = game_type
        self.num_games = num_games

        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[AZ_NodeBatchResponse]               = in_queue
        self.metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]]= metrics_queue

        self.out_queue:     mpQueueGen[list[AZ_NodeBatchRequest] | None]   = out_queue
        self.output_games:  mpQueueGen[tuple[str, GameBase[GameStateT]]]   = output_games

        # Consider below zipped by thread index
        self.thread_inbox:  list[AZ_SimulationReturnType | None]  = []
        self.threads:       list[Thread]                          = []
        self.inbox_cv:      list[Condition]                       = []

        # Metrics
        self.total_moves = 0
        self.sent_batches = 0
        self.sent_requests = 0
        self.received_batches = 0
        self.received_results = 0
        self.active_games = 0
        # self.thread_total_moves: list[int] = [0] * self.num_games

    def run(self) -> None:
        self.metrics_lock = Lock()
        self.outbound_cv = Condition()

        self.staging_out_queue: Queue[AZ_NodeBatchRequest] = Queue()
        Thread(target=self.inbound_mailman_d, args=[], daemon=True).start()
        Thread(target=self.outbound_mailman_d, args=[], daemon=True).start()
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
            batch_response = self.in_queue.get()
            assert batch_response.worker_id == self.worker_id
            for result in batch_response.results:
                thread_id, eval_result = result
                assert thread_id >= 0 and thread_id < len(self.threads)
                assert self.thread_inbox[thread_id] is None

                # Open (with mutex) the right thread's mailbox, insert the
                # received BatchResponse and notify the waiting thread.
                self.inbox_cv[thread_id].acquire()
                self.thread_inbox[thread_id] = eval_result
                self.inbox_cv[thread_id].notify()
                self.inbox_cv[thread_id].release()

            with self.metrics_lock:
                self.received_batches += 1
                self.received_results += len(batch_response.results)

    def outbound_mailman_d(self) -> None:
        """
        Consolidates individual thread requests into a single batch request to send to the MP world.
        Only send when:
            - at least 40% of active threads have requests, or
            - when the first request has been waiting for MAX_WAIT_S seconds.
        """
        requests: list[AZ_NodeBatchRequest] = []
        first_request_time = 0.0
        while True:
            if not requests:
                request = self.staging_out_queue.get()
                first_request_time = time.perf_counter()
                requests.append(request)
            
            # Drain the staging queue for more requests to batch together.
            while True:
                try:
                    request = self.staging_out_queue.get(block=False)
                except Empty:
                    break
                requests.append(request)

            # Send the batch request to the worker pool.
            if len(requests) >= max(1, int(self.WAIT_FRAC * self.active_games)) or (time.perf_counter() - first_request_time) >= self.MAX_WAIT_S:
                self.out_queue.put(requests)
                with self.metrics_lock:
                    self.sent_batches += 1
                    self.sent_requests += len(requests)
                requests = []
            else:
                # If we don't have enough requests to send a batch, wait for more.
                time.sleep(self.RECHECK_INTERVAL_S)
                continue

    def metrics_d(self) -> None:
        while True:
            time.sleep(self.metrics_interval_s)
            self.metrics_queue.put(
                (self.worker_id,
                 self.total_moves,
                 self.sent_batches,
                 self.sent_requests,
                 self.received_batches,
                 self.received_results)
            )

    def run_game(self, thread_id: int, game: GameBase[GameStateT]) -> None:
        logging.info(f"thread {thread_id} started")

        # player
        first_player = game.current_player
        while True:
            logging.debug(f"thread {thread_id} move {len(game.state_history)}")

            # Game after last move is still going, create new MCTS instance.
            MCTS_instance = self.MCTS_factory.make_instance(game=game)
            while MCTS_instance.root.visits <= MCTS_instance.rollouts:
                # While not enough rollouts:
                logging.debug(f"thread {thread_id} {MCTS_instance.root.visits} visits, not enough")
                # Make next batch
                logging.debug(f"{MCTS_instance.root=}")
                node, request_or_response = MCTS_instance.one_round_batch(self.worker_id, thread_id)

                if isinstance(request_or_response, AZ_NodeBatchRequest):
                    # Send batch request to worker pool.
                    self.staging_out_queue.put(request_or_response)

                    # Get or wait (block thread) for the result.
                    self.inbox_cv[thread_id].acquire()
                    # NOTE: Is this while loop strictly needed??
                    while self.thread_inbox[thread_id] is None:
                        self.inbox_cv[thread_id].wait()
                    eval_result = self.thread_inbox[thread_id]
                    self.thread_inbox[thread_id] = None
                    self.inbox_cv[thread_id].release()
                    assert eval_result is not None

                    y_policy, y_value = eval_result
                    # Update prior probability from the neural network output
                    node.expand(y_policy)
                    node.backpropogate(1, y_value, False)
                else:
                    # Cached result; no request is sent.
                    logging.debug(f"thread {thread_id} (cached) batchresponse")
                    node.backpropogate(1, request_or_response[1], True)

                MCTS_instance.root.print_children(0, limit=2)

            # Make move
            root = MCTS_instance.root
            action_probs = np.zeros(self.game_type.action_size, dtype=np.float32)
            for child in root.children:
                action_probs[child.parent_action] = child.visits
            action_probs /= np.sum(action_probs, dtype=np.float32)

            game.state_history.append(game.state.neutral_state(game.current_player))
            game.action_prob_history.append(action_probs)

            action = np.random.choice(self.game_type.action_size, p=action_probs)
            game.make_move(action)

            logging.debug(f"thread {thread_id} made move {action}")
            with self.metrics_lock:
                self.total_moves += 1

            value, terminated = game.get_value_and_terminated(action)
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
