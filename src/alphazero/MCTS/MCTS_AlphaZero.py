from __future__ import annotations
from dataclasses import dataclass
import logging
import math
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

import chess

from alphazero.batching.NodeBatch import BatchSparseLegalPolicy, SparseLegalPolicy, AZ_NodeRequest
from alphazero.games.GameBase import GameBase
from alphazero.games.GameStateBase import GameStateBase
# from alphazero.games.Chess import ChessGame
# from alphazero.games.ChessUtils import decode_action, encode_move

"""
Special Implementation of MCTS for AlphaZero chess.
Instead of copying the game state into each node, we only store the game state in the root node.
When we traverse down the tree, we generate the next game state on-the-fly by applying the action that led to the child node.
On the way back up the tree we pop the game state back to the root node.
"""

GameStateT = TypeVar('GameStateT', bound='GameStateBase')
MoveT = TypeVar("MoveT")

class MCTS_Factory(Generic[GameStateT, MoveT]):
    DEFAULT_EXPLORATION_PARAM = 1.41
    debug = 0
    exploration = DEFAULT_EXPLORATION_PARAM
    temperature = 1.0
    DIRICHLET_CONST = 10
    dirichlet_epsilon = 0.25
    fast_rollouts_chance = 0.75
    fast_rollouts_factor = 0.25

    def __init__(self, rollouts: int) -> None:
        self.debug = 0
        self.rollouts = rollouts

    def set_debug_state(self, debug: int) -> None:
        self.debug = debug
        self.__class__.debug = debug

    def set_exploration_param(self, exploration: float) -> None:
        self.exploration = exploration
        self.__class__.exploration = exploration

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature
        self.__class__.temperature = temperature

    def make_instance(self, **kwargs) -> MCTS_Instance[GameStateT, MoveT]:
        return MCTS_Instance(self.rollouts, MCTS_factory=self, **kwargs)

@dataclass(slots=True, frozen=True)
class MCTS_Result(Generic[GameStateT, MoveT]):
    action_stats: list[tuple[float, float, float, int, MoveT, float]]
    best_move: MoveT
    state: GameStateT
    # board: chess.Board

    @property
    def best_action(self) -> int:
        return self.state.encode_move(self.best_move)

    @property
    def best_move_str(self) -> str:
        return self.state.move_repr(self.best_move)

    def __str__(self):
        return "\n".join(
            f"{pct_visits:>6.1%} ({visits:>4}) visits | E(value): {avg_value:+.2f} ({avg_value/2 + 0.5:>6.1%})"
            f" | ucb {ucb:.3f} | prior {prior:.2f} move {self.state.move_repr(move)}{" <<<" if move == self.best_move else ""}"
            for pct_visits, avg_value, ucb, visits, move, prior in self.action_stats
        )

class MCTS_Instance(Generic[GameStateT, MoveT]):
    # Create a new MCTS from current state (player +1 us/-1 them)
    def __init__(self, rollouts: int, *, MCTS_factory: MCTS_Factory[GameStateT, MoveT], **kwargs) -> None:
        assert isinstance(kwargs['game'], GameBase), "MCTS_Instance requires a 'game' argument of type GameBase"
        self.game: GameBase[GameStateT] = kwargs['game']

        player = self.game.current_player
        assert player == -1 or player == 1

        self.game_state = self.game.state.__copy__()
        self.model = kwargs.get('model', None)

        # Recover from a previous MCTS tree if possible, instead of starting from scratch.
        if kwargs.get('root', None) is not None:
            self.root: Node[GameStateT, MoveT] = kwargs['root']
            assert isinstance(self.root, Node)
            assert self.root.parent is None
            assert self.root.parent_move is None
        else:
            self.root = Node(None, None, player)
        self.rollouts = rollouts
        self.MCTS_factory = MCTS_factory

    def num_rollouts(self, fast: bool=False) -> int:
        if fast:
            return int(self.rollouts * self.MCTS_factory.fast_rollouts_factor)
        else:
            return self.rollouts

    def one_round_batch(self, thread_id: int) -> \
                            tuple[Node, AZ_NodeRequest] | \
                            tuple[Node, SparseLegalPolicy]:
        """
        Returns a leaf node and a request for the model to evaluate the leaf node's state.
        This state is encoded in the neutral player's perspective.
        """
        # Selection:
        # Get to a leaf node. (A leaf is any non-terminal node i.e. has potential
        # children that aren't made yet.)
        # If not currently a leaf node, traverse to child of current
        # which maximises UCB score.
        curr = self.root
        while not curr.is_leafnode():
            curr = curr.select(self.game_state)

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
            logging.debug(f"at node {curr.parent_move}")

        if curr.value is not None:
            if self.MCTS_factory.debug >= 2:
                logging.debug(f"using cache value {curr.value}")
            return curr, SparseLegalPolicy.value_only(curr.value)

        if curr.parent_move is not None:
            reward, terminated = self.game_state.get_value_and_terminated()
            if terminated:
                # reward describes the player who just moved;
                # curr.player is that player's opponent.
                value = -float(reward)
                return curr, SparseLegalPolicy.value_only(value)

        return curr, AZ_NodeRequest(
            thread_id,
            curr.player,
            self.game_state.get_legal_actions(),
            self.game_state.encode_for_model(curr.player)
        )

    def _search(self) -> None:
        import torch

        assert self.model is not None, "Model must be set before calling search()"
        while self.root.visits <= self.rollouts:
            node, request_or_response = self.one_round_batch(0)

            if isinstance(request_or_response, AZ_NodeRequest):
                self.model.eval()
                with torch.no_grad():
                    x = torch.as_tensor(request_or_response.state, device="cuda", dtype=torch.float32).unsqueeze(0)
                    legal_masks = torch.as_tensor(request_or_response.legal_mask, device="cuda", dtype=torch.bool).unsqueeze(0)

                    policy_logits, values = self.model(x)

                    _, sparse_legal_policy = BatchSparseLegalPolicy.from_dense(
                        legal_masks=legal_masks,
                        logits=policy_logits,
                        values=values,
                        thread_idxs=np.array([0])
                    ).unpack()[0]

                node.expand(sparse_legal_policy)
                node.backpropogate(1, sparse_legal_policy.value, False, self.game_state)
            else:
                # Cached/terminal result; no request is sent.
                node.backpropogate(1, request_or_response.value, True, self.game_state)

    def search(self, state: GameStateT) -> MCTS_Result[GameStateT, MoveT]:
        self._search()
        children_details: list[tuple] = []
        assert self.root.legal_actions is not None
        assert self.root.priors is not None

        for idx, child in enumerate(self.root.children):
            if child is not None:
                assert child.parent_move is not None
                children_details.append((
                    child.visits / self.root.visits if self.rollouts else math.nan,
                    (-child.value_sum / child.visits) if child.visits else math.nan,
                    self.root.get_ucb(idx),
                    child.visits,
                    child.parent_move,
                    float(self.root.priors[idx])
                ))
            else:
                # Unexpanded child
                children_details.append((
                    0,                      # pct_visits
                    math.nan,               # avg_value
                    self.root.get_ucb(idx), # ucb
                    0,                      # visits
                    state.decode_action(int(self.root.legal_actions[idx]), self.root.player),
                    float(self.root.priors[idx])                # prior_prob
                ))

        best_action = max(children_details, key=lambda x: x[0])[4]
        return MCTS_Result(children_details, best_action, self.game_state)
        # return MCTS_Result(children_details, best_action, self.game_state.board.copy(stack=True))

    # def search_concise(self) -> SparseLegalPolicy:
    #     self._search()

    #     action_probs = np.zeros(self.game.action_size, dtype=np.float32)
    #     legal_actions: list[int] = []

    #     for child in self.root.children:
    #         assert child.parent_action is not None
    #         action_idx = encode_move(self.chess_state.board, child.parent_action, self.game.current_player)
    #         action_probs[action_idx] = child.visits
    #         legal_actions.append(action_idx)

    #     action_probs /= np.sum(action_probs, dtype=np.float32)
    #     legal_actions_arr = np.array(legal_actions, dtype=np.uint16)
    #     sparse_probs = action_probs[legal_actions_arr].astype(np.float32)

    #     return SparseLegalPolicy(
    #             legal_actions=legal_actions_arr,
    #             probs=sparse_probs,
    #             value=self.root.value_sum / self.root.visits if self.root.visits > 0 else 0.0
    #         )

    def dispose(self, kept_child: Node | None=None) -> None:
        """Release this tree except for child identified by chosen_action"""
        if kept_child is not None:
            if not any(child is kept_child for child in self.root.children):
                raise ValueError("kept_child must be a direct child of root")
            kept_child.parent = None
            kept_child.parent_move = None

        pending = [self.root]

        while pending:
            node = pending.pop()

            if node is kept_child:
                continue  # Preserve this entire subtree.

            pending.extend((child for child in node.children if child is not None))
            node.children.clear()
            node.parent = None

    def get_child_by_action(self, action: int) -> Node:
        # Find the child node corresponding to the chosen action
        result = None
        for child in self.root.children:
            if child is not None and child.parent_move is not None:
                action_idx = encode_move(self.game_state.board, child.parent_move, self.game.current_player)
                if action_idx == action:
                    result = child
                    break

        if result is None:
            raise ValueError(f"Chosen action {action} does not correspond to any child of the root.")
        return result

class Node(Generic[GameStateT, MoveT]):
    __slots__ = ["parent",
                 "parent_move",
                 "player",
                 "children",
                 "legal_actions",
                 "priors",
                 "value",
                #  "prior_prob",
                 "value_sum",
                 "visits"]
    def __init__(self,
                 parent: Node[GameStateT, MoveT] | None,
                 parent_move: MoveT | None,
                #  prior_prob: float,
                 player: int) -> None:
        if parent is None:
            assert parent_move is None
        else:
            assert parent_move is not None

        self.parent = parent                # Parent MCTS node
        self.parent_move = parent_move      # Move made by parent node to get to here
        self.player = player                # Player to move

        # Child MCTS nodes
        self.children: list[Node[GameStateT, MoveT] | None]  = []
        # Sparse array of legal actions
        self.legal_actions: NDArray[np.uint16] | None = None
        # Child priors corresponding to legal_actions
        self.priors: NDArray[np.float32] | None       = None

        # If terminal node, assign value on first simulation and return it
        self.value: float | None                      = None
        self.value_sum: float                         = 0
        self.visits: int                              = 0

    def is_leafnode(self) -> bool:
        return self.legal_actions is None

    def select(self, state: GameStateT) -> Node[GameStateT, MoveT]:
        """
        Side-effect: pushes the move of the best child onto the board.
        """
        assert self.legal_actions is not None
        assert self.priors is not None
        assert len(self.legal_actions) > 0

        best_child_index = max(
            range(len(self.legal_actions)),
            key=self.get_ucb,
        )

        best_child = self.children[best_child_index]
        if best_child is None:
            best_child = Node(
                parent=self,
                parent_move=state.decode_action(int(self.legal_actions[best_child_index]), self.player),
                player=-self.player
            )
            self.children[best_child_index] = best_child

        assert best_child.parent_move is not None
        state.push_to_board(best_child.parent_move)
        return best_child

    def get_ucb(self, index: int) -> float:
        assert self.priors is not None
        child = self.children[index]

        visits = 0 if child is None else child.visits
        q = (
            -child.value_sum / visits
            if child is not None and visits > 0
            else 0.0
        )
        u = (
            MCTS_Factory.exploration
            * float(self.priors[index])
            * math.sqrt(self.visits)
            / (1 + visits)
        )

        return q + u

    def expand(self, sparse_policy: SparseLegalPolicy) -> None:
        assert self.legal_actions is None
        assert self.priors is None
        assert np.all(
            (sparse_policy.probs >= 0.0)
            & (sparse_policy.probs <= 1.0)
        ), "Sparse policy probabilities must be in range [0, 1]"

        self.legal_actions = sparse_policy.legal_actions.copy()
        self.priors = sparse_policy.probs.copy()
        self.children = [None] * len(self.legal_actions)

    def backpropogate(self, visits: int, total_value: float, terminal: bool, state: GameStateT) -> None:
        self.value_sum += total_value
        self.visits += visits
        if terminal:
            single_value = total_value / visits
            if (self.value is not None):
                # Already set, check consistency for a terminal node
                assert math.isclose(single_value, self.value), "Inconsistent value for terminal node."
            else:
                self.value = single_value
        if self.parent is not None:
            assert state.pop_board() == self.parent_move, "Board state does not match parent action during backpropagation."
            self.parent.backpropogate(visits, -1 * total_value, False, state)

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
            index = self.parent.children.index(self)
            ucb = self.parent.get_ucb(index)
            move = self.parent_move

            logging.debug(f"{'\t' * depth}"
                        f"{move}: {pct_visits:>6.1%} ({visits:>4}) visits | E(value): {avg_value:+.2f} ({avg_value/2 + 0.5:>6.1%})"
                        f" | ucb {ucb:.3f}")
        if depth >= limit:
            return

        for child in self.children:
            if child is not None:
                child.print_children(depth + 1, limit=limit)
            else:
                logging.debug(f"{'\t' * (depth + 1)} <unexpanded child>")
