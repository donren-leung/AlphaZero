from __future__ import annotations
import logging
import math
import random
from copy import copy
from dataclasses import dataclass

from TicTacToe import TicTacToeGame, TicTacToeState

import numpy as np

class MCTS_Factory():
    DEFAULT_EXPLORATION_PARAM = 1.41

    def __init__(self, rollouts: int) -> None:
        self.exploration: float = self.DEFAULT_EXPLORATION_PARAM
        self.debug = 0
        self.rollouts = rollouts

    def set_debug_state(self, debug: int) -> None:
        self.debug = debug

    def set_exploration_param(self, exploration: float) -> None:
        self.exploration = exploration

    def make_instance(self, *args) -> MCTS_Instance:
        return MCTS_Instance(self.rollouts, *args, MCTS_factory=self)

@dataclass
class MCTS_Result():
    action_stats: list[tuple[float, float, float, int, int]]
    best_action: int

    def __str__(self):
        return "\n".join(
            f"{pct_visits:>6.1%} ({visits:>4}) visits | E(value): {-avg_value:+.2f} ({-avg_value/2 + 0.5:>6.1%})"
            f" | ucb {ucb:.3f} | move {move}{" <<<" if move == self.best_action else ""}"
            for pct_visits, avg_value, ucb, visits, move in self.action_stats
        )

class MCTS_Instance():
    # Create a new MCTS from current state (player +1 us/-1 them)
    def __init__(self, rollouts: int, state: TicTacToeState, player: int, *,
                 MCTS_factory: MCTS_Factory) -> None:
        assert player == -1 or player == 1
        self.rollouts = rollouts
        self.root = Node(None, None, None, state, player, MCTS_factory=MCTS_factory)
        self.MCTS_factory = MCTS_factory

    # Do MCTS and return the visit counts of the root children
    def search(self) -> MCTS_Result:
        self.root.expand()

        for _ in range(self.rollouts):
            self.one_round_mcts()

        logging.debug("finished MCTS")
        if self.MCTS_factory.debug >= 1:
            self.root.print_children(limit=4)

        children_details = [(
                child.visits / self.rollouts,
                child.value_sum / child.visits,
                self.root.get_ucb(child),
                child.visits,
                child.parent_action if child.parent_action is not None else -1)
            for child in self.root.children
        ]
        best_action = max(children_details, key=lambda x: x[0])[4]
        return MCTS_Result(children_details, best_action)

    def one_round_mcts(self) -> None:
        # Selection:
        # Get to a leaf node. (A leaf is any non-terminal node i.e. has potential
        # children that aren't made yet.)
        # If not currently a leaf node, traverse to child of current
        # which maximises UCB score.
        curr = self.root
        if self.MCTS_factory.debug >= 2:
            curr.print_children(0)
        while not curr.is_leafnode():
            curr = curr.select()

        # (Now at a leaf node)
        # Expansion
        # Has a rollout been played from this leaf before (n > 0)?
        # If not, do rollout from this node
        # If yes (n = 1) for each available action, add a new child node to tree
        # rollout from a random child.
        # If the game is ended at this point, obviously there can't be children
        # so just "simulate" and record the value.
        if curr.visits == 0 or curr.is_terminal:
            simulating_node = curr
        else:
            curr.expand()
            simulating_node = random.choice(curr.children)

        # Simulation
        # Take random actions until terminated
        value = simulating_node.simulate()

        # Backpropogation
        # Update scores and tally up the tree
        simulating_node.backpropogate(value)

class Node():
    def __init__(self, parent: 'Node' | None, parent_action: int | None,
                 value: int | None, state: TicTacToeState, player: int,
                 *, MCTS_factory: MCTS_Factory) -> None:
        assert (
            (parent is None and parent_action is None) or
            (parent is not None and parent_action is not None)
        )
        self.parent = parent                # Parent MCTS node
        self.parent_action = parent_action  # Move made by parent node to get to here

        self.children: list[Node] = []      # Child MCTS nodes
        self.state = state                  # Game representation
        self.player = player                # Player to move

        # If terminal node don't simulate, just return the value
        self.value: int | None = value
        self.value_sum: float = 0
        self.visits: int = 0
        self.MCTS_factory = MCTS_factory

    @property
    def is_terminal(self) -> bool:
        return self.value is not None

    def is_leafnode(self) -> bool:
        if self.visits <= 1 and self.parent is not None:
            assert len(self.children) == 0, f"Visits = {self.visits}, node should not be expanded yet."
        elif self.is_terminal:
            return True
        else:
            assert len(self.children) > 0, \
                f"Visits = {self.visits}, node has no children but node should have been expanded (check if it's terminal)."

        return len(self.children) == 0

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
        q = 1 - (child.value_sum / child.visits + 1) / 2
        return q + self.MCTS_factory.exploration * math.sqrt(math.log(self.visits) / child.visits)

    def expand(self) -> None:
        assert len(self.children) == 0, f"Node already has children: {self.children}."
        if self.parent is not None:
            assert self.visits == 1, f"Non-root node has {self.visits} visits; it should be 1."

        curr_state = self.state
        valid_actions = curr_state.get_legal_actions()
        for action_idx in np.flatnonzero(valid_actions):
            new_state = curr_state.get_next_state(action_idx, self.player)

            value, terminated = new_state.get_value_and_terminated(action_idx)
            if not terminated:
                value = None
            else:
                value = value * -1

            self.children.append(Node(self, action_idx, value, new_state, -1 * self.player,
                                      MCTS_factory=self.MCTS_factory))

    def simulate(self) -> float:
        assert self.parent_action is not None
        if self.is_terminal:
            assert self.value is not None
            return float(self.value)

        curr_state = copy(self.state)
        curr_player = self.player
        while True:
            valid_action_indexes = np.flatnonzero(curr_state.get_legal_actions())
            next_action = random.choice(valid_action_indexes)

            curr_state = curr_state.get_next_state(next_action, curr_player, copy=False)
            value, terminated = curr_state.get_value_and_terminated(next_action)
            if terminated:
                if self.MCTS_factory.debug >= 2:
                    logging.debug(f"==terminated: v {value} p {curr_player}")
                    logging.debug("\n" + str(curr_state))
                return value if curr_player == self.player else -value

            curr_player = -1 * curr_player

    def backpropogate(self, value: float) -> None:
        self.value_sum += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropogate(-1 * value)

    def print_children(self, depth: int=0, *, limit: int=1) -> None:
        if not self.MCTS_factory.debug:
            return
        if depth == 0:
            logging.debug("@@@ printing children")

        if self.parent is None:
            visits = self.visits
            avg_value = self.value_sum / (self.visits + 0.01)
            logging.debug(f"{'\t' * depth} ({visits:>4}) visits | E(value): {-avg_value:+.2f} ({-avg_value/2 + 0.5:>6.1%})")
        else:
            pct_visits = self.visits / (self.parent.visits + 0.01)
            visits = self.visits
            avg_value = self.value_sum / (self.visits + 0.01)
            ucb = self.parent.get_ucb(self)
            move = self.parent_action

            logging.debug(f"{'\t' * depth}"
                        f"{move}: {pct_visits:>6.1%} ({visits:>4}) visits | E(value): {-avg_value:+.2f} ({-avg_value/2 + 0.5:>6.1%})"
                        f" | ucb {ucb:.3f}")
        if depth >= limit:
            return

        for child in self.children:
            child.print_children(depth + 1, limit=limit)
