import logging
import sys

from math import nan
from multiprocessing.context import BaseContext
from queue import Queue, Empty
from threading import Thread, local, Condition
from typing import Generic, TypeVar, Type

import numpy.typing as npt

from .NodeBatch import AZ_NodeBatchRequest, AZ_NodeBatchResponse, AZ_SimulationReturnType
from .Pool import mpQueueGen
from alphazero.games.GameBase import GameBase
from alphazero.MCTS.MCTS_AlphaZero import MCTS_Factory, Node

GameT = TypeVar('GameT', bound='GameBase')

class GameWorker(Generic[GameT], object):
    """
    A CPU assigned to send batches of nodes to the evaluator for
    each position, for each parallel game of specified game type.
    """
    def __init__(self, game_type: type[GameT], *,
                 num_games: int, output_games: mpQueueGen[tuple[str, GameT]],
                 in_queue: mpQueueGen[AZ_NodeBatchResponse],
                 out_queue: mpQueueGen[list[AZ_NodeBatchRequest] | None],
                 worker_id: int, MCTS_factory: MCTS_Factory):
        self.game_type = game_type
        self.num_games = num_games
        # self.finished_game_list: Queue[GameT] = Queue()
        self.worker_id = worker_id
        self.MCTS_factory = MCTS_factory

        # Threading
        # Multiprocess stuff
        self.in_queue:      mpQueueGen[AZ_NodeBatchResponse]               = in_queue

        self.out_queue:     mpQueueGen[list[AZ_NodeBatchRequest] | None]   = out_queue
        self.output_games:  mpQueueGen[tuple[str, GameT]]                  = output_games

        # Consider below zipped by thread index
        self.thread_inbox:  list[AZ_SimulationReturnType | None]  = []
        self.threads:       list[Thread]                       = []
        self.inbox_cv:      list[Condition]                    = []

    def run(self) -> None:
        self.staging_out_queue: Queue[AZ_NodeBatchRequest] = Queue()
        Thread(target=self.inbound_mailman_d, args=[], daemon=True).start()
        Thread(target=self.outbound_mailman_d, args=[], daemon=True).start()
        
        for i in range(self.num_games):
            new_game_instance = self.game_type()
            new_inbox_cv = Condition()

            t = Thread(target=self.run_game, args=[i, new_game_instance])
            t.name = f"w{self.worker_id}.g{i}"

            self.thread_inbox.append(None)
            self.threads.append(t)
            self.inbox_cv.append(new_inbox_cv)

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

    def outbound_mailman_d(self) -> None:
        """
        Consolidates individual thread requests into a single batch request to send to the MP world.
        """
        while True:
            request = self.staging_out_queue.get()
            requests: list[AZ_NodeBatchRequest] = [request]
            
            # Drain the staging queue for more requests to batch together.
            while True:
                try:
                    request = self.staging_out_queue.get(block=False)
                except Empty:
                    break
                requests.append(request)

            # Send the batch request to the worker pool.
            self.out_queue.put(requests)            

    def run_game(self, thread_id: int, game: GameT) -> None:
        logging.info(f"thread {thread_id} started")
        while True:
            # Check if game has ended (which can't happen on first move)
            # Need to make this assumption due to the arguments required.
            if len(game.action_history) == 0:
                pass
            else:
                _, terminated = game.get_value_and_terminated(game.action_history[-1])
                if terminated:
                    break

            logging.debug(f"thread {thread_id} iteration {len(game.action_history)}: {game.action_history}")
            # Game after last move is still going, create new MCTS instance.
            MCTS_instance = self.MCTS_factory.make_instance(game=game)

            while MCTS_instance.root.visits < MCTS_instance.rollouts:
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
