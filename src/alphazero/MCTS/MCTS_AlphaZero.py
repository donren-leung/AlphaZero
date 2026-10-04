from __future__ import annotations
import logging
import math

from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from alphazero.batching.NodeBatch import AZ_NodeRequest, AZ_NodeBatchResponse, AZ_SimulationReturnType
from alphazero.games.GameBase import GameBase
from alphazero.games.GameStateBase import GameStateBase

# from viztracer import log_sparse

"""
Batched, Parallel Child MCTS Implementation for parallel games.

MCTS_Instance.one_round_batch() used by GameWorker to select a leaf node and create a NodeBatchRequest for all children of the leaf node.
simulate_() used by CPUPool.

Note: Cannot be called by main.py because .search() is not implemented. Use MCTS_multichild.py instead for a single game.
"""

class MCTS_Factory(object):
    DEFAULT_EXPLORATION_PARAM = 1.41
    debug = 0
    exploration = DEFAULT_EXPLORATION_PARAM

    def __init__(self, rollouts: int) -> None:
        self.debug = 0
        self.rollouts = rollouts

    def set_debug_state(self, debug: int) -> None:
        self.debug = debug
        self.__class__.debug = debug

    def set_exploration_param(self, exploration: float) -> None:
        self.exploration = exploration
        self.__class__.exploration = exploration

    def make_instance(self, **kwargs) -> MCTS_Instance:
        return MCTS_Instance(self.rollouts, MCTS_factory=self, **kwargs)

@dataclass(slots=True, frozen=True)
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
    def __init__(self, rollouts: int, *, MCTS_factory: MCTS_Factory, **kwargs) -> None:
        game: GameBase = kwargs['game']
        assert isinstance(game, GameBase)
        self.game = game

        player = game.current_player
        assert player == -1 or player == 1

        state = game.state
        assert isinstance(state, GameStateBase)

        # TODO: recover from a previous MCTS tree if possible, instead of starting from scratch.
        self.root = Node(None, None, 1.0, state, player)
        self.rollouts = rollouts
        self.MCTS_factory = MCTS_factory

    def one_round_batch(self, worker_id: int, thread_id: int) -> \
                            tuple[Node, AZ_NodeRequest] | \
                            tuple[Node, AZ_SimulationReturnType]:
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
            # so just sample the value.
        # If not:
        #   **Different in AZ**:
        #   Evaluate leaf node in network to get its value.
        #   Then finally populate its children, since network also gets priors.
        if self.MCTS_factory.debug >= 2:
            logging.debug(f"at node {curr.parent_action}")

        if curr.value is not None:
            if self.MCTS_factory.debug >= 2:
                logging.debug(f"using cache value {curr.value}")
            return curr, (np.empty(0, dtype=np.float32), curr.value)

        if curr.parent_action is not None:
            reward, terminated = curr.state.get_value_and_terminated(
                curr.parent_action
            )
            if terminated:
                # reward describes the player who just moved;
                # curr.player is that player's opponent.
                value = -float(reward)
                return curr, (np.empty(0, dtype=np.float32), value)

        return curr, AZ_NodeRequest(
            worker_id,
            thread_id,
            curr.player,
            curr.state.neutral_state(curr.player)
        )

    def search(self, model: torch.nn.Module) -> tuple[npt.NDArray[np.float32], float]:
        """
        Non-batched MCTS search. For use in a single game.
        """
        while self.root.visits <= self.rollouts:
            node, request_or_response = self.one_round_batch(0, 0)

            if isinstance(request_or_response, AZ_NodeRequest):
                model.eval()
                with torch.no_grad():
                    tensor_state = request_or_response.state.to_tensor().unsqueeze(0).to("cuda").to(torch.float32)
                    policy, value = model(tensor_state)

                    policy = torch.softmax(policy, dim=1).squeeze(0).detach().cpu().numpy()
                    value = value.item()

                node.expand(policy)
                node.backpropogate(1, value, False)
            else:
                # Cached/terminal result; no request is sent.
                node.backpropogate(1, request_or_response[1], True)

        action_probs = np.zeros(self.game.action_size, dtype=np.float32)
        for child in self.root.children:
            action_probs[child.parent_action] = child.visits
        action_probs /= np.sum(action_probs, dtype=np.float32)
        value = self.root.value_sum / self.root.visits if self.root.visits > 0 else 0.0

        return action_probs, value

class Node(object):
    __slots__ = ["parent",
                 "parent_action",
                 "children",
                 "state",
                 "player",
                 "value",
                 "prior_prob",
                 "value_sum",
                 "visits"]
    def __init__(self,
                 parent: Node | None,
                 parent_action: int | None,
                 prior_prob: float,
                 state: GameStateBase,
                 player: int) -> None:
        if parent is None:
            assert parent_action is None
        else:
            assert parent_action is not None

        self.parent = parent                # Parent MCTS node
        self.parent_action = parent_action  # Move made by parent node to get to here

        self.children: list[Node] = []      # Child MCTS nodes
        self.state = state                  # Game representation
        self.player = player                # Player to move

        # If terminal node, assign value on first simulation and return it
        self.value: float | None = None
        # Prior probability of this node. Not initialised until the network returns.
        self.prior_prob: float = prior_prob

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
        assert isinstance(child.prior_prob, float) and 0.0 <= child.prior_prob <= 1.0
        q = (-child.value_sum / child.visits + 1) / 2 if child.visits else 0.0
        u = (
            MCTS_Factory.exploration
            * child.prior_prob
            * math.sqrt(self.visits)
            / (1 + child.visits)
        )

        return q + u

    def expand(self, policy: npt.NDArray[np.float32]) -> None:
        assert len(self.children) == 0, f"Node already has children: {self.children}."
        # if self.parent is not None:
        #     assert self.visits == 1, f"Non-root node has {self.visits} visits; it should be 1."

        curr_state = self.state
        valid_actions = curr_state.get_legal_actions(self.player)
        # assert valid actions and policy are the same length
        assert len(valid_actions) == len(policy), f"valid_actions and policy must be the same length, got {len(valid_actions)} and {len(policy)}."
        # assert policy is a valid probability distribution
        assert np.isclose(np.sum(policy), 1.0), f"policy must sum to 1, got {np.sum(policy)}."
        # assert policy is non-negative
        assert np.all(policy >= 0), f"policy must be non-negative, got {policy}."

        norm_policy = mask_and_norm(valid_actions, policy)
        for action_idx in np.flatnonzero(valid_actions):
            action_idx = int(action_idx)
            new_state = curr_state.get_next_state(action_idx, self.player)
            child = Node(
                parent=self,
                parent_action=action_idx,
                prior_prob=float(norm_policy[action_idx]),
                state=new_state,
                player=-1 * self.player
            )
            self.children.append(child)

    def simulate(self, executor: ProcessPoolExecutor, pending_simulations: dict[Future, Node],
                 *, target_sims: int) -> None:
        raise DeprecationWarning()

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

def mask_and_norm(valid: npt.NDArray[np.bool_], policy: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    policy *= valid
    policy_sum = np.sum(policy)
    if policy_sum == 0:
        # If all actions are invalid, make all actions equally probable
        return np.full_like(policy, 1.0 / len(policy), dtype=np.float32)
    else:
        return policy / policy_sum
