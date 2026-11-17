import logging
import time

from queue import Queue, Empty
from threading import Thread, Lock, Condition
from typing import Generic, Self, TypeVar, Type, cast

import numpy as np

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeRequest, AZ_NodeBatchResponse, BatchSparseLegalPolicy, SparseLegalPolicy
from .Queues import mpQueueGen
from alphazero.games.GameBase import GameBase
from alphazero.games.GameStateBase import GameStateBase
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory, Node

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
GameT = GameBase[GameStateT]

class GameWorker(Generic[GameStateT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position, for each parallel game of specified game type.
    """
    metrics_interval_s = 0.5
    MAX_WAIT_S = 20.0
    SMALL_BATCH_SIZE = 4
    MAX_WAIT_S_SMALL = 0.1

    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 256

    def __init__(self, game_type: type[GameBase[GameStateT]], *,
                 num_games: int, output_games: mpQueueGen[tuple[str, GameBase[GameStateT]]],
                 in_queue: mpQueueGen[AZ_NodeBatchResponse],
                 out_queue: mpQueueGen[AZ_NodeBatchRequest | None],
                 metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]],
                 worker_id: int, MCTS_factory: MCTS_Factory,
                 random_seed: int):
        self.game_type = game_type
        self.num_games = num_games

        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory
        self.random_seed = random_seed

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[AZ_NodeBatchResponse]               = in_queue
        self.metrics_queue: mpQueueGen[tuple[int, int, int, int, int, int]]= metrics_queue

        self.out_queue:     mpQueueGen[AZ_NodeBatchRequest | None]         = out_queue
        self.output_games:  mpQueueGen[tuple[str, GameBase[GameStateT]]]   = output_games

        # Consider below zipped by thread index
        self.thread_inbox:  list[tuple[
            BatchSparseLegalPolicy, int, int, int] | None
        ]                                                   = []
        self.threads:       list[Thread]                    = []
        self.inbox_cv:      list[Condition]                 = []

        # Metrics
        self.total_moves = 0
        self.sent_batches = 0
        self.sent_requests = 0
        self.received_batches = 0
        self.received_results = 0
        self.active_games = 0

    def run(self) -> None:
        self.metrics_lock = Lock()
        self.outbound_cv = Condition()

        self.staging_out_queue: Queue[AZ_NodeRequest] = Queue()
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
        'Mailman' thread -- Gets AZ_NodeBatchResponse from the MP world and
        in the local process unpacks it and puts each game's result into the correct per-thread mailbox.
        """
        while True:
            # Blocking call to get a BatchResponse from the MP queue.
            batch_response = self.in_queue.get()
            assert batch_response.worker_id == self.worker_id

            batch = batch_response.batch_sparse_legal_policy
            offset = 0
            for row, (thread_idx, length) in enumerate(zip(
                cast(list[int], batch.thread_idxs.tolist()),
                cast(list[int], batch.lengths.tolist()),
                strict=True,
            )):
                end = offset + length
                # Open (with mutex) the right thread's mailbox, insert the
                # received BatchResponse and notify the waiting thread.
                self.inbox_cv[thread_idx].acquire()
                assert self.thread_inbox[thread_idx] is None
                self.thread_inbox[thread_idx] = (batch, offset, end, row)
                self.inbox_cv[thread_idx].notify()
                self.inbox_cv[thread_idx].release()

                offset = end

            with self.metrics_lock:
                self.received_batches += 1
                self.received_results += len(batch_response)

            del batch, batch_response

    def calculate_batch_size(self) -> int:
        if self.active_games <= 0:
            return 0
        elif self.active_games <= self.SMALL_BATCH_SIZE:
            return 1

        # 1/4 of active games, rounded down to nearest power of 2
        batch_size = 1 << (self.active_games.bit_length() - 2)
        batch_size = max(1, batch_size // 2)

        return max(self.MIN_BATCH_SIZE, min(batch_size, self.MAX_BATCH_SIZE))

    def outbound_mailman_d(self) -> None:
        """
        Consolidates individual thread requests. Stacks arrays to send as a single request to the MP world.
        Only send when:
            - Enough requests are waiting to fill a batch, or
            - Over MAX_WAIT_S seconds have passed since the first request in the batch was received.
        """
        requests: list[AZ_NodeRequest] = []
        first_request_time = 0.0
        while True:
            if not requests:
                request = self.staging_out_queue.get()
                first_request_time = time.perf_counter()
                requests.append(request)
            
            # Drain the staging queue for more requests to batch together.
            target_batch_size = max(1, self.calculate_batch_size())
            while len(requests) < target_batch_size:
                if target_batch_size <= self.SMALL_BATCH_SIZE:
                    remaining = self.MAX_WAIT_S_SMALL - (time.perf_counter() - first_request_time)
                else:
                    remaining = self.MAX_WAIT_S - (time.perf_counter() - first_request_time)
                if remaining <= 0:
                    break

                try:
                    request = self.staging_out_queue.get(timeout=remaining)
                    requests.append(request)
                except Empty:
                    break

            # Send full or timed-out batch to the MP world.
            payload = AZ_NodeBatchRequest.from_NodeRequests(self.worker_id, requests[:target_batch_size])
            self.out_queue.put(payload)
            with self.metrics_lock:
                self.sent_batches += 1
                self.sent_requests += len(payload.curr_players)

            requests = requests[target_batch_size:]
            if requests:
                first_request_time = time.perf_counter()

    def metrics_d(self) -> None:
        while True:
            time.sleep(self.metrics_interval_s)
            with self.metrics_lock:
                self.metrics_queue.put(
                    (self.worker_id,
                    self.total_moves,
                    self.sent_batches,
                    self.sent_requests,
                    self.received_batches,
                    self.received_results)
                )

    # TODO: might migrate to MCTS
    def search_move(self,
                    thread_id: int,
                    game: GameBase[GameStateT],
                    initial_root: Node | None,
                    rng: np.random.Generator):
        MCTS_instance = self.MCTS_factory.make_instance(game=game, root=initial_root)
        try:
            while MCTS_instance.root.visits <= MCTS_instance.num_rollouts(initial_root is not None):
                # While not enough rollouts:
                logging.debug(f"thread {thread_id} {MCTS_instance.root.visits} visits, not enough")
                # Make next batch
                logging.debug(f"{MCTS_instance.root=}")
                node, request_or_response = MCTS_instance.one_round_batch(thread_id)

                if isinstance(request_or_response, AZ_NodeRequest):
                    # Send batch request to worker pool.
                    self.staging_out_queue.put(request_or_response)

                    # Get or wait (block thread) for the result.
                    self.inbox_cv[thread_id].acquire()
                    # NOTE: Is this while loop strictly needed??
                    while self.thread_inbox[thread_id] is None:
                        self.inbox_cv[thread_id].wait()
                    delivery = self.thread_inbox[thread_id]
                    self.thread_inbox[thread_id] = None
                    self.inbox_cv[thread_id].release()

                    assert delivery is not None
                    batch, start, end, row = delivery

                    legal_actions = batch.legal_actions[start:end]
                    probs = batch.probs[start:end]
                    value = float(batch.values[row])

                    # Introduce Dirichlet noise to root node for exploration
                    # but only if in full search mode (want to maximise strength in fast)
                    if node is MCTS_instance.root and initial_root is None:
                        num_legal = legal_actions.shape[0]
                        alpha = self.MCTS_factory.DIRICHLET_CONST / num_legal
                        noise = rng.dirichlet(np.full(num_legal, alpha)).astype(np.float32)

                        probs = (
                            (1.0 - self.MCTS_factory.dirichlet_epsilon) * probs
                            + self.MCTS_factory.dirichlet_epsilon * noise
                        ).astype(np.float32, copy=False)

                    # Update prior probability from the neural network output
                    node.expand(legal_actions, probs)
                    node.backpropogate(1, value, False, MCTS_instance.game_state)

                    del delivery, batch, legal_actions, probs
                else:
                    # Cached result; no request is sent.
                    logging.debug(f"thread {thread_id} (cached) batchresponse")
                    node.backpropogate(1, request_or_response.value, True, MCTS_instance.game_state)

                # if self.MCTS_factory.debug > 0:
                #     MCTS_instance.root.print_children(0, limit=2)

            # Extract final MCTS statistics
            root = MCTS_instance.root
            assert root.ints is not None
            legal_actions_arr = root.ints[0]
            sparse_probs = root.ints[1].astype(np.float32)
            sparse_probs /= sparse_probs.sum(dtype=np.float32)

            # If we did a full search i.e. not fast, provided root was None then
            # we append to history.
            if initial_root is None:
                game.history['states'].append(game.state.to_compact_repr(game.current_player))
                game.history['MCTS_results'].append(SparseLegalPolicy(
                    legal_actions=legal_actions_arr,
                    probs=sparse_probs,
                    value=root.value_sum / root.visits if root.visits > 0 else 0.0
                ))
                game.history['player'].append(game.current_player)

            return legal_actions_arr, sparse_probs, MCTS_instance
        except Exception as e:
            MCTS_instance.dispose(kept_child=None)
            logging.error(f"thread {thread_id} exception: {e}")
            raise

    def run_game(self, thread_id: int, game: GameBase[GameStateT]) -> None:
        logging.info(f"thread {thread_id} started")
        rng = np.random.default_rng([self.random_seed, self.worker_id, thread_id])

        first_player = game.current_player
        initial_root = None
        while True:
            logging.debug(f"thread {thread_id} move {len(game.history.get('states', []))}")

            # Game after last move is still going, create new MCTS instance.
            legal_actions_arr, sparse_probs, MCTS_instance = self.search_move(
                thread_id,
                game,
                initial_root=initial_root,
                rng=rng
            )

            temperature_probs = np.power(sparse_probs, 1.0 / self.MCTS_factory.temperature)
            temperature_probs /= np.sum(temperature_probs, dtype=np.float32)
            action = int(rng.choice(legal_actions_arr, p=temperature_probs))

            # Next move is randomly a fast (reuse) or slow (restart) MCTS search.
            # Based on https://arxiv.org/pdf/1902.10565v5 (KataGo) "Playout Cap Randomization"
            # If fast, keep previous root; dispose of all child nodes except for the chosen one.
            if rng.random() < self.MCTS_factory.fast_rollouts_chance:
                initial_root = MCTS_instance.get_child_by_action(action)
                MCTS_instance.dispose(kept_child=initial_root)
            else:
                initial_root = None
                MCTS_instance.dispose(kept_child=None)

            game.make_move(action)

            logging.debug(f"thread {thread_id} made move {action}")
            with self.metrics_lock:
                self.total_moves += 1

            value, terminated = game.get_value_and_terminated()
            if terminated:
                assert len(game.history['states']) == len(game.history['MCTS_results']) == len(game.history['player'])
                hist_player = first_player
                last_player = game.get_opponent(game.current_player)

                for hist_player in game.history['player']:
                    outcome = value if hist_player == last_player else -value
                    game.history['outcome'].append(outcome)

                to_clear = [initial_root]
                while to_clear:
                    node = to_clear.pop()
                    if node is not None:
                        to_clear.extend((child for child in node.children if child is not None))
                        node.children.clear()
                        node.parent = None

                del MCTS_instance
                break

            del MCTS_instance

        # Game finished
        # TODO: use pipe?
        logging.info(f"thread {thread_id} finished game")
        self.output_games.put((f"{self.worker_id}_{thread_id}", game))

        with self.metrics_lock:
            self.active_games -= 1
