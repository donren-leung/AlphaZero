import logging
import time

from queue import Queue, Empty
from threading import Thread, Lock, Condition, Event
from typing import Generic, Self, TypeVar, Type, cast

import numpy as np

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeRequest, AZ_NodeBatchResponse, BatchSparseLegalPolicy, SparseLegalPolicy
from .Queues import mpQueueGen
from .Metrics import WorkerMetrics, WorkerTotals
from alphazero.games.GameBase import GameBase
from alphazero.games.GameStateBase import CompactRepr, GameStateBase
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory, Node

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
GameT = GameBase[GameStateT]

type CompletedGame = tuple[str, list[tuple[CompactRepr, SparseLegalPolicy, int]]]


class GameWorker(Generic[GameStateT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position, for each parallel game of specified game type.
    """
    metrics_interval_s = 0.5
    MAX_WAIT_S = 20.0
    SMALL_BATCH_SIZE = 4
    MAX_WAIT_S_SMALL = 0.02

    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 256

    def __init__(self, game_type: type[GameBase[GameStateT]], *,
                 num_games: int, output_games: mpQueueGen[CompletedGame],
                 in_queue: mpQueueGen[AZ_NodeBatchResponse],
                 out_queue: mpQueueGen[AZ_NodeBatchRequest | None],
                 metrics_queue: mpQueueGen[WorkerMetrics],
                 worker_id: int, MCTS_factory: MCTS_Factory,
                 random_seed: int,
                 iteration: int):
        self.game_type = game_type
        self.num_games = num_games

        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory
        self.random_seed = random_seed
        self.iteration = iteration

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[AZ_NodeBatchResponse]               = in_queue
        self.metrics_queue = metrics_queue

        self.out_queue:     mpQueueGen[AZ_NodeBatchRequest | None]         = out_queue
        self.output_games:  mpQueueGen[CompletedGame]   = output_games

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
        self.fast_moves = 0
        self.fast_skipped_rounds = 0
        self.fast_terminal_rounds = 0
        self.fast_budget_rounds = 0
        self.full_moves = 0
        self.full_cached_evals = 0
        self.full_terminal_evals = 0
        self.full_requested_evals = 0
        self.active_games = 0

    def run(self) -> None:
        self.metrics_lock = Lock()
        self.outbound_cv = Condition()

        self.staging_out_queue: Queue[AZ_NodeRequest] = Queue()
        Thread(target=self.inbound_mailman_d, args=[], daemon=True).start()
        Thread(target=self.outbound_mailman_d, args=[], daemon=True).start()
        games_finished = Event()
        metrics_thread = Thread(target=self.metrics_d, args=(games_finished,), daemon=True)
        metrics_thread.start()

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
        for thread in self.threads:
            thread.join()
        games_finished.set()
        metrics_thread.join()

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

    def metrics_d(self, games_finished: Event) -> None:
        while True:
            finished = games_finished.wait(self.metrics_interval_s)
            with self.metrics_lock:
                totals = WorkerTotals(
                    moves=self.total_moves, sent_batches=self.sent_batches, sent_requests=self.sent_requests,
                    received_batches=self.received_batches, received_results=self.received_results,
                    fast_moves=self.fast_moves, fast_skipped_rounds=self.fast_skipped_rounds,
                    fast_terminal_rounds=self.fast_terminal_rounds, fast_budget_rounds=self.fast_budget_rounds,
                    full_moves=self.full_moves,
                    full_cached_evals=self.full_cached_evals, full_terminal_evals=self.full_terminal_evals,
                    full_requested_evals=self.full_requested_evals,
                )
            self.metrics_queue.put(WorkerMetrics(self.worker_id, totals))
            if finished:
                return

    def _evaluate_leaf(self, request: AZ_NodeRequest) -> SparseLegalPolicy:
        # Send batch request to worker pool.
        thread_id = request.thread_idx

        # Get or wait (block thread) for the result.
        self.staging_out_queue.put(request)
        with self.inbox_cv[thread_id]:
            while self.thread_inbox[thread_id] is None:
                self.inbox_cv[thread_id].wait()
            delivery = self.thread_inbox[thread_id]
            self.thread_inbox[thread_id] = None

        assert delivery is not None
        batch, start, end, row = delivery
        return SparseLegalPolicy(
            batch.legal_actions[start:end], batch.probs[start:end], batch.values[row].item()
        )

    def search_move(self,
                    thread_id: int,
                    game: GameBase[GameStateT],
                    previous_root: Node | None,
                    rng: np.random.Generator, *,
                    fast: bool):
        # If fast, continue search from the previous MCTS tree.
        # Otherwise, if previous_root is given, we may use that for cached evals.
        cache_root = None if fast else previous_root
        MCTS_instance = self.MCTS_factory.make_instance(game=game, root=previous_root if fast else None)

        try:
            # A fresh root needs one expansion round before rounds can visit its children.
            round_budget = 1 + MCTS_instance.num_rollouts(fast)
            cached_evals = requested_evals = terminal_evals = 0
            while MCTS_instance.root.visits < round_budget:
                node, result = MCTS_instance.one_round_batch(thread_id, cache_root)
                if isinstance(result, AZ_NodeRequest):
                    result = self._evaluate_leaf(result)
                    requested_evals += 1
                # else, result is a SparseLegalPolicy from a terminal node or cached evaluation.
                elif result.legal_actions.size == 0:
                    # terminal nodes can be backpropagated immediately; cached ones need to be expanded first.
                    node.backpropogate(1, result.value, True, MCTS_instance.game_state)
                    terminal_evals += 1
                    continue
                else:
                    cached_evals += 1

                # Introduce Dirichlet noise to root node for exploration
                # but only if in full search mode (want to maximise strength in fast). In other words:
                # Fast mode inherits a bit of the dirichlet noise from previous root through previously done work.
                # Full mode draws its own noise.
                probs = result.probs
                if node is MCTS_instance.root and not fast:
                    alpha = self.MCTS_factory.DIRICHLET_CONST / probs.size
                    noise = rng.dirichlet(np.full(probs.size, alpha)).astype(np.float32)
                    epsilon = self.MCTS_factory.dirichlet_epsilon
                    probs = ((1.0 - epsilon) * probs + epsilon * noise).astype(np.float32)

                node.expand(result.legal_actions, probs, result.value)
                node.backpropogate(1, result.value, False, MCTS_instance.game_state)
                # Release views into the shared response batch before waiting for the next one.
                del probs, result

            # Extract final MCTS statistics
            root = MCTS_instance.root
            assert root.ints is not None
            legal_actions_arr = root.ints[0]
            sparse_probs = root.ints[1].astype(np.float32)
            sparse_probs /= sparse_probs.sum(dtype=np.float32)

            rounds_run = cached_evals + terminal_evals + requested_evals
            if not fast:
                assert rounds_run == round_budget, f"Full search counted {rounds_run} of {round_budget} rounds"
                game.history['states'].append(game.state.to_compact_repr(game.current_player))
                game.history['MCTS_results'].append(SparseLegalPolicy(
                    legal_actions=legal_actions_arr,
                    probs=sparse_probs,
                    value=root.value_sum / root.visits if root.visits > 0 else 0.0
                ))
                game.history['player'].append(game.current_player)

            with self.metrics_lock:
                if fast:
                    self.fast_moves += 1
                    self.fast_budget_rounds += round_budget
                    self.fast_terminal_rounds += terminal_evals
                    self.fast_skipped_rounds += round_budget - rounds_run
                else:
                    self.full_moves += 1
                    self.full_cached_evals += cached_evals
                    self.full_terminal_evals += terminal_evals
                    self.full_requested_evals += requested_evals

            return legal_actions_arr, sparse_probs, MCTS_instance
        except Exception as e:
            MCTS_instance.dispose(kept_child=None)
            logging.error(f"thread {thread_id} exception: {e}")
            raise
        finally:
            if cache_root is not None:
                cache_root.clear_subtree()

    def run_game(self, thread_id: int, game: GameBase[GameStateT]) -> None:
        logging.info(f"thread {thread_id} started")
        rng = np.random.default_rng([self.iteration, self.random_seed, self.worker_id, thread_id])

        fast = False
        previous_root = None
        while True:
            logging.debug(f"thread {thread_id} move {len(game.history.get('states', []))}")

            # Game after last move is still going, create new MCTS instance.
            legal_actions_arr, sparse_probs, MCTS_instance = self.search_move(
                thread_id,
                game,
                previous_root=previous_root,
                rng=rng,
                fast=fast,
            )

            temperature_probs = np.power(sparse_probs, 1.0 / self.MCTS_factory.temperature)
            temperature_probs /= np.sum(temperature_probs, dtype=np.float32)
            action = int(rng.choice(legal_actions_arr, p=temperature_probs))

            game.make_move(action)

            logging.debug(f"thread {thread_id} made move {action}")
            with self.metrics_lock:
                self.total_moves += 1

            value, terminated = game.get_value_and_terminated()
            if terminated:
                MCTS_instance.dispose()
                assert len(game.history['states']) == len(game.history['MCTS_results']) == len(game.history['player'])
                last_player = game.get_opponent(game.current_player)

                for hist_player in game.history['player']:
                    outcome = value if hist_player == last_player else -value
                    game.history['outcome'].append(outcome)

                break

            # Based on https://arxiv.org/pdf/1902.10565v5 (KataGo) "Playout Cap Randomization"
            # Next move is randomly a fast (reuse) or slow (use previous tree as cache) MCTS search.
            # Keep previous root; dispose of all child nodes except for the chosen one.
            fast = rng.random() < self.MCTS_factory.fast_rollouts_chance
            previous_root = MCTS_instance.get_child_by_action(action)
            MCTS_instance.dispose(kept_child=previous_root)
            del MCTS_instance

        # Game finished
        # TODO: use pipe?
        logging.info(f"thread {thread_id} finished game")
        examples = list(zip(game.history["states"], game.history["MCTS_results"],
                            game.history["outcome"], strict=True))
        self.output_games.put((f"{self.worker_id}_{thread_id}", examples))

        with self.metrics_lock:
            self.active_games -= 1
