from typing import Dict

from cdrl.agent.ordering.ordering_state import OrderingState


class OrderingNode(object):
    """
    MCTS tree node for the variable-ordering search.

    Attributes:
        state:    The partial ordering at this node.
        N:        Visit count.
        W:        Total accumulated reward (Q = W/N).
        children: Mapping from action (variable index) to child OrderingNode.
    """
    __slots__ = ("state", "N", "W", "children")

    def __init__(self, state):
        self.state = state
        self.N = 0
        self.W = 0.0
        self.children = {}  # type: Dict[int, OrderingNode]

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def update(self, reward):
        self.N += 1
        self.W += reward
