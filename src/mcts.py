"""
Logic for Monte Carlo Tree Search
"""

import logging
from typing import List, Optional
from c4 import N_COLS, Pos, get_legal_moves, is_game_over, make_move

import numpy as np

from nn import EvaluatePos, Policy


class Node:
    """Represents a node in the MCTS tree."""

    pos: Pos
    parent: Optional["Node"]
    visit_count: int
    exploitation_value_sum: float
    initial_policy_value: float
    children: Optional[List["Node"]]

    def __init__(
        self,
        pos: Pos,
        parent: Optional["Node"] = None,
        initial_policy_value: float = 0.0,
    ):
        self.pos = pos
        self.parent = parent
        self.visit_count = 0
        self.exploitation_value_sum = 0
        self.initial_policy_value = initial_policy_value
        self.children = None

    def exploitation_value(self) -> float:
        """The exploitation component of the UTC value, i.e. the average win rate."""
        return self.exploitation_value_sum / self.visit_count

    def exploration_value(self) -> float:
        """
        The exploration component of the UCT value. Higher visit counts result in lower values.
        """
        if self.parent is None:
            parent_visit_count = self.visit_count
        else:
            parent_visit_count = self.parent.visit_count

        return (
            np.sqrt(np.log(parent_visit_count) / (self.visit_count + 1))
            * self.initial_policy_value
        )

    def uct_value(self, exploration_constant: float) -> float:
        """
        The UCT value of this node. Represents the lucrativeness of this node according to MCTS.
        """
        return (
            self.exploitation_value() + exploration_constant * self.exploration_value()
        )

    def expand_children(self, evaluate_pos: EvaluatePos) -> None:
        """Expands the children of this leaf node and backpropagates the value up the tree."""
        logger = logging.getLogger(__name__)

        assert self.children is None, "expand_children() called on a node with children"

        # If the game is over, simply backpropagate the objective game outcome
        terminal_value = is_game_over(self.pos)
        if terminal_value is not None:
            self._backpropagate(terminal_value.value)
            return

        policy, value = evaluate_pos(self.pos)

        # Renormalize policy after masking with legal moves
        legal_moves = get_legal_moves(self.pos)
        policy &= legal_moves
        if np.sum(policy) == 0.0:
            # It may be possible for the policy to be all zeros after masking with legal moves.
            # In this case, we set the policy to be uniform over legal moves.
            logger.warning(
                "Zero policy after masking with legal moves. Using uniform policy."
            )
            policy = legal_moves
        policy /= np.sum(policy)

        self.children = [
            Node(
                pos=make_move(self.pos, move),
                parent=self,
                initial_policy_value=policy[move],
            )
            if legal_moves[move]
            else None
            for move in range(N_COLS)
        ]
        self._backpropagate(value)

    def _backpropagate(self, value: float) -> None:
        """Backpropagate value up the tree, alternating value signs for each step."""
        pos = self
        while pos is not None:
            pos.visit_count += 1
            pos.exploitation_value_sum += value
            value = -value
            pos = pos.parent


def _select_leaf(node: Node, exploration_constant: float) -> Node:
    """
    Selects a leaf node or terminal node from the tree according to the UCT value.
    """

    def sort_key(n: Optional[Node]) -> float:
        if n is None:
            return float("-inf")
        return n.uct_value(exploration_constant)

    while node.children is not None:
        node = max(node.children, key=sort_key)

    return node


def mcts(
    pos: Pos, n_iterations: int, exploration_constant: float, evaluate_pos: EvaluatePos
) -> Policy:
    """
    Runs the MCTS algorithm to determine the best move from the given position.
    Returns a policy vector indicating the probability of playing in each column.
    """
    root = Node(pos)
    for _ in range(n_iterations):
        leaf = _select_leaf(root, exploration_constant)
        leaf.expand_children(evaluate_pos)

    child_visits = np.array(
        [child.visit_count if child is not None else 0 for child in root.children]
    )
    return child_visits / np.sum(child_visits)
