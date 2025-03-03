from __future__ import annotations
import logging
import math
import random
import time

from copy import copy
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass

from games.GameBase import GameBase
from games.GameStateBase import GameStateBase

from viztracer import log_sparse
import numpy as np

class MCTS_Factory(object):
    DEFAULT_EXPLORATION_PARAM = 1.41
    debug = 0
    exploration = DEFAULT_EXPLORATION_PARAM

    def __init__(self, rollouts: int, multi_sims: int) -> None:
        self.exploration: float = self.DEFAULT_EXPLORATION_PARAM
        self.debug = 0
        self.rollouts = rollouts
        # When simulating a node, how many times to simulate?
        self.multi_sims = multi_sims

    def set_debug_state(self, debug: int) -> None:
        self.debug = debug
        self.__class__.debug = debug

    def set_exploration_param(self, exploration: float) -> None:
        self.exploration = exploration
        self.__class__.exploration = exploration

    def make_instance(self, *args) -> MCTS_Instance:
        return MCTS_Instance(self.rollouts, self.multi_sims, *args, MCTS_factory=self)

@dataclass
class MCTS_Result():
    action_stats: list[tuple[float, float, float, int, int]]
    best_action: int

    def __str__(self):
        return "\n".join(
            f"{pct_visits:>6.1%} ({visits:>4}) visits | E(value): {avg_value:+.2f} ({avg_value/2 + 0.5:>6.1%})"
            f" | ucb {ucb:.3f} | move {move}{" <<<" if move == self.best_action else ""}"
            for pct_visits, avg_value, ucb, visits, move in self.action_stats
        )

class MCTS_Instance(object):
    # Create a new MCTS from current state (player +1 us/-1 them)
    def __init__(self, rollouts: int, multi_sims: int, state: GameStateBase, player: int, *,
                 MCTS_factory: MCTS_Factory) -> None:
        assert player == -1 or player == 1
        self.rollouts = rollouts
        self.root = Node(None, None, state, player)
        self.MCTS_factory = MCTS_factory
        self.multi_sims = multi_sims

    # Do MCTS and return the visit counts of the root children
    def search(self) -> MCTS_Result:
        # self.root.expand()
        with ProcessPoolExecutor(max_workers=10) as executor:
            pending_simulations: dict[Future[SimulationReturnType], Node] = {}

            while self.root.visits <= self.rollouts:
                if self.MCTS_factory.debug >= 2:
                    self.root.print_children(0, limit=2)
                self.one_round_mcts(executor, pending_simulations)
                # time.sleep(1e-5)

            # for future in as_completed(pending_simulations):
            #     result = future.result()
            #     pending_simulations[future].backpropogate(*result)

        logging.debug("finished MCTS")
        if self.MCTS_factory.debug >= 1:
            self.root.print_children(limit=3)

        children_details = [(
                child.visits / self.root.visits if self.rollouts else math.nan,
                (child.value_sum / child.visits) if child.visits else math.nan,
                self.root.get_ucb(child),
                child.visits,
                child.parent_action if child.parent_action is not None else -1)
            for child in self.root.children
        ]
        best_action = max(children_details, key=lambda x: x[0])[4]
        return MCTS_Result(children_details, best_action)

    def one_round_mcts(self, executor: ProcessPoolExecutor,
                       pending_simulations: dict[Future[SimulationReturnType], Node]) -> None:
        # Selection:
        # Get to a leaf node. (A leaf is any non-terminal node i.e. has potential
        # children that aren't made yet.)
        # If not currently a leaf node, traverse to child of current
        # which maximises UCB score.
        curr = self.root
        while not curr.is_leafnode():
            curr = curr.select()

        # (Now at a leaf node)
        # Expansion:
        # Is this node terminal?
        # If yes:
            # Obviously we can't make more children
            # so just sample the value * multi_sims.
        # If not:
        #   for each available action, add a new child node to tree.
        #   rollout from all children.
        if self.MCTS_factory.debug >= 2:
            logging.debug(f"at node {curr.parent_action}")

        if curr.value is not None:
            if self.MCTS_factory.debug >= 2:
                logging.debug(f"using cache value {curr.value} * {self.MCTS_factory.multi_sims}")
            curr.backpropogate(self.MCTS_factory.multi_sims,
                               self.MCTS_factory.multi_sims * curr.value, False)
            return
        else:
            curr.expand()
            for child in curr.children:
                # Simulation of a child:
                # Take random actions until terminated
                # This call should be non-blocking as much as possible.
               child.simulate(executor, pending_simulations,
                              target_sims=self.multi_sims)

        # Wait for all children simulations to be completed
        for future in as_completed(pending_simulations):
            node = pending_simulations.pop(future)
            result = future.result()
            node.backpropogate(*result, stop_at_node=node.parent)

class Node():
    def __init__(self, parent: 'Node' | None, parent_action: int | None,
                 state: GameStateBase, player: int) -> None:
        assert (
            (parent is None and parent_action is None) or
            (parent is not None and parent_action is not None)
        )
        self.parent = parent                # Parent MCTS node
        self.parent_action = parent_action  # Move made by parent node to get to here

        self.children: list[Node] = []      # Child MCTS nodes
        self.state = state                  # Game representation
        self.player = player                # Player to move

        # If terminal node, assign value on first simulation and return it
        self.value: float | None = None
        self.value_sum: float = 0
        self.visits: int = 0

    def is_leafnode(self) -> bool:
        if len(self.children) > 0:
            return False
        else:
            return True

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: Node) -> float:
        assert child is not None
        if child.visits == 0:
            return np.inf
        return ((child.value_sum / child.visits + 1) / 2) + \
            MCTS_Factory.exploration * math.sqrt(math.log(self.visits) / child.visits)

    def expand(self) -> None:
        assert len(self.children) == 0, f"Node already has children: {self.children}."
        # if self.parent is not None:
        #     assert self.visits == 1, f"Non-root node has {self.visits} visits; it should be 1."

        curr_state = self.state
        valid_actions = curr_state.get_legal_actions()
        for action_idx in np.flatnonzero(valid_actions):
            new_state = curr_state.get_next_state(action_idx, self.player)
            self.children.append(Node(self, action_idx, new_state, -1 * self.player))

    def simulate(self, executor: ProcessPoolExecutor, pending_simulations: dict[Future, Node],
                 *, target_sims: int) -> None:
        assert self.parent_action is not None
        assert self.visits == 0
        assert self.value is None

        curr_state = self.state
        curr_player = self.player

        # Include parent action to find out if this node is terminal
        future = executor.submit(simulate_,
                                 curr_state,
                                 curr_player,
                                 self.parent_action,
                                 target_sims=target_sims,
                                 debug=MCTS_Factory.debug)

        pending_simulations[future] = self

    def backpropogate(self, visits: int, total_value: float, terminal: bool,
                      *, stop_at_node: Node | None=None) -> None:
        self.value_sum += total_value
        self.visits += visits
        if terminal:
            single_value = total_value / visits
            if (self.value is not None):
                # Already set, check consistency for a terminal node
                assert math.isclose(single_value, self.value), "Inconsistent value for terminal node."
            else:
                self.value = single_value
        if self is not stop_at_node and self.parent is not None:
            self.parent.backpropogate(visits, -1 * total_value, False)

    def print_children(self, depth: int=0, *, limit: int=1) -> None:
        if depth == 0:
            logging.debug("@@@ printing children")

        if self.parent is None:
            visits = self.visits
            avg_value = self.value_sum / (self.visits + 0.01)
            logging.debug(f"{'\t' * depth} ({visits:>4}) visits | E(value): {avg_value:+.2f} ({avg_value/2 + 0.5:>6.1%})")
        else:
            pct_visits = self.visits / (self.parent.visits + 0.01)
            visits = self.visits
            avg_value = self.value_sum / (self.visits + 0.01)
            ucb = self.parent.get_ucb(self)
            move = self.parent_action

            logging.debug(f"{'\t' * depth}"
                        f"{move}: {pct_visits:>6.1%} ({visits:>4}) visits | E(value): {avg_value:+.2f} ({avg_value/2 + 0.5:>6.1%})"
                        f" | ucb {ucb:.3f}")
        if depth >= limit:
            return

        for child in self.children:
            child.print_children(depth + 1, limit=limit)

# visits, total value, terminal
SimulationReturnType = tuple[int, float, bool]

# Returns a value, with respect to the node that called the simulation.
# The value is the expected score for the parent node taking an action which
# resulted in the calling-node.
def simulate_(curr_state: GameStateBase, curr_player: int, parent_action: int,
              *, target_sims: int, debug: int=0) -> tuple[int, float, bool]:
    # Find out if the game is already over
    value, terminated = curr_state.get_value_and_terminated(parent_action)
    if terminated:
        # Game is ended for the current player, return 0/+1 (for the parent who made the action).
        return (target_sims, target_sims * value, True)

    # TODO: Is copying necessary?
    origin_curr_state = copy(curr_state)
    origin_curr_player = curr_player

    value_sum = 0.0
    for _ in range(target_sims):
        curr_player = origin_curr_player
        curr_state = copy(origin_curr_state)
        while True:
            valid_action_indexes = np.flatnonzero(curr_state.get_legal_actions())
            next_action = random.choice(valid_action_indexes)

            curr_state = curr_state.get_next_state(next_action, curr_player, copy=False)
            value, terminated = curr_state.get_value_and_terminated(next_action)
            curr_player = -1 * curr_player

            if terminated:
                # if debug >= 2:
                #     logging.debug(f"==terminated: v {-value} for cp {curr_player} sp {simulating_player}")
                #     logging.debug("\n" + str(curr_state))

                # Game is ended for the current player.
                # If the player-to-move was the calling-node player,
                # return 0/+1 (for the parent who made the action to the calling-node)
                # otherwise, flip.
                value_sum += (value if curr_player == origin_curr_player else -value)
                break

    return (target_sims, value_sum, False)
