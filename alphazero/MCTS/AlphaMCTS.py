from __future__ import annotations
import logging
import math
import random

from copy import copy
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass

from AlphaZero.alphazero.batching.AlphaNodeResult import NodeRequest, NodeResponse
from games.GameBase import GameBase
from games.GameStateBase import GameStateBase

import asyncio

# from viztracer import log_sparse
import numpy as np
import torch
import torch.nn.functional as F

###
# This iteration calls the AlphaZero neural network to get the policy and value for each node.
# In the future, it implements virtual loss for parallel traversing of the same tree.
# Also, from the node of the next move, saves the tree for the next MCTS so we can reuse it.
###

class MCTS_Factory(object):
    DEFAULT_EXPLORATION_PARAM = 1.41
    debug = 0
    exploration = DEFAULT_EXPLORATION_PARAM

    def __init__(self, rollouts: int, processes: int) -> None:
        self.debug = 0
        self.rollouts = rollouts
        self.processes = processes

    def set_debug_state(self, debug: int) -> None:
        self.debug = debug
        self.__class__.debug = debug

    def set_exploration_param(self, exploration: float) -> None:
        self.exploration = exploration
        self.__class__.exploration = exploration

    def make_instance(self, **kwargs) -> MCTS_Instance:
        return MCTS_Instance(self.rollouts, self.multi_sims, self.processes,
                             MCTS_factory=self, **kwargs)

@dataclass(slots=True, frozen=True)
class MCTS_Result():
    # pct_visits, avg_value, ucb, visits, move
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
    def __init__(self, rollouts: int, multi_sims: int, processes: int,
                 *, MCTS_factory: MCTS_Factory, **kwargs,
                 ) -> None:
        game: GameBase = kwargs['game']
        assert issubclass(game.__class__, GameBase)

        player = game.current_player
        assert player == -1 or player == 1

        state = game.state
        assert issubclass(state.__class__, GameStateBase)

        assert processes >= 1

        self.root = Node(None, None, state, player)
        self.rollouts = rollouts
        self.multi_sims = multi_sims
        self.processes = processes
        self.MCTS_factory = MCTS_factory

    async def do_one_traversal(self, worker_id: int, thread_id: int) -> \
                            tuple[list[Node], NodeRequest] | \
                            tuple[Node, NodeResponse]:
        # Selection:
        # Get to a leaf node. (A leaf is any node that hasn't been expanded yet, i.e. has no children.)
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
            # so just backpropagate the cached value (saved on prior visit).
        # If not:
        #   Call the neural network for p, v.
        #   Assign the value v to the node.
        #   For each available action, add a new child node to tree.
        #   assign each child its policy from parent node.

        # Backpropagation:
        # (Not implemented here): update the value and visit count for each node on the path from the leaf to the root.
        if self.MCTS_factory.debug >= 2:
            logging.debug(f"at node {curr.parent_action}")

        if curr.value is not None:
            if self.MCTS_factory.debug >= 2:
                logging.debug(f"using cache value {curr.value}")
            curr.backpropogate(curr.value)
        else:
            await policy, value = asyncio.sleep(0)  # Placeholder for async call to neural network
            policy = F.softmax(policy, dim=1).cpu().numpy()
            valid_moves = curr.state.get_legal_actions(curr.player)
            policy *= valid_moves
            policy /= np.sum(policy)

            value = value.item()

            curr.expand(policy)
            curr.backpropogate(curr.value)

class Node(object):
    __slots__ = ["parent",
                 "parent_action",
                 "children",
                 "state",
                 "player",
                 "value",
                 "value_sum",
                 "visits"]
    def __init__(self, parent: 'Node' | None, parent_action: int | None,
                 state: GameStateBase, player: int, prior: float) -> None:
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
        self.prior = prior
        self.value: float | None = None
        self.value_sum: float = 0
        self.visits: int = 0

    def is_leafnode(self) -> bool:
        if len(self.children) > 0:
            return False
        else:
            return True

    def select(self) -> Node:
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        assert best_child is not None
        return best_child

    def get_ucb(self, child: Node) -> float:
        assert child is not None
        if child.visits == 0:
            q_value = 0
        else:
            q_value = ((child.value_sum / child.visits + 1) / 2)
        return q_value + MCTS_Factory.exploration * math.sqrt(self.visits / (child.visits + 1)) * child.prior

    def expand(self, policy) -> None:
        assert len(self.children) == 0, f"Node already has children: {self.children}."
        # if self.parent is not None:
        #     assert self.visits == 1, f"Non-root node has {self.visits} visits; it should be 1."
        curr_state = self.state
        for action_idx, prob in enumerate(policy):
            if prob > 0:
                new_state = curr_state.get_next_state(action_idx, self.player)
                self.children.append(Node(self, action_idx, new_state, -1 * self.player, prob))

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
            self.parent.backpropogate(visits, -1 * total_value, False, stop_at_node=stop_at_node)

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

