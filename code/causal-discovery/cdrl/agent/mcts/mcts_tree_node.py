"""
MCTSTreeNode: tree node used by the edge-based UCT agent (MCTSAgent).

Each node represents a partially constructed DAG - the state reached after a
sequence of edge add/remove decisions in the edge MDP. The node tracks visit
counts, Q-value estimates (updated with a moving average), and which actions
have been tried so far, plus optional predictor values that can bias expansion
toward promising actions.
"""

import numpy as np
from cdrl.utils.config_utils import local_seed


class MCTSTreeNode(object):
    """
    A single node in the UCT search tree for the edge-based causal discovery MDP.

    `N` and `Q` accumulate statistics during tree search: Q is updated as a
    running mean (Q = Q + (R - Q) / N) rather than a sum, so it always holds
    the average reward over all backpropagated simulations through this node.

    `predictor_values` are set at expansion time and used to bias the choice of
    which unvisited action to expand next (proportional sampling). They are uniform
    (all 1.0) in the current implementation - the field exists to support a learned
    prior in future.
    """
    def __init__(self, state, parent_nodes, parent_action, valid_actions, depth=-1, remaining_budget=-1.):
        self.state = state
        self.parent_nodes = parent_nodes
        self.parent_action = parent_action

        if self.parent_nodes is None:
            self.depth = depth
        else:
            self.depth = self.parent_nodes[0].depth + 1

        self.remaining_budget = remaining_budget

        self.children = {}

        self.N = 0.0
        self.Q = 0.0

        self.valid_actions = np.array(valid_actions)
        self.num_valid_actions = len(valid_actions)
        self.actions_arr_index = {self.valid_actions[i]: i for i in range(self.num_valid_actions)}
        self.visited_actions = np.full(self.num_valid_actions, False, dtype=np.bool)

        # Will be initialized when node is expanded.
        self.predictor_values = None

    def assign_predictor_values(self, predictor_values):
        """Sets predictor values for all children of a node."""
        self.predictor_values = predictor_values

    def get_predictor_value(self, action):
        """Retrieves the predictor value for the node corresponding to the given action."""
        action_idx = self.actions_arr_index[action]
        return self.predictor_values[action_idx]

    def choose_action(self, random_seed):
        """
        Samples an unvisited action, weighted by predictor values.

        Falls back to uniform sampling when predictor values sum to zero or
        contain NaNs (e.g., all-zero uniform prior).
        """
        remaining_actions = self.valid_actions[~self.visited_actions]
        action_priors = self.predictor_values[~self.visited_actions]

        with np.errstate(divide='ignore', invalid='ignore'):
            action_priors = np.divide(action_priors, np.sum(action_priors))

        with local_seed(random_seed):
            if not np.isnan(action_priors).any():
                chosen_action = np.random.choice(remaining_actions, p=action_priors)
            else:
                chosen_action = np.random.choice(remaining_actions)

        chosen_idx = self.actions_arr_index[chosen_action]
        self.visited_actions[chosen_idx] = True
        return chosen_action

    def update_estimates(self, R):
        """Updates Q-estimates with a moving average."""
        self.N += 1
        self.Q = self.Q + ((R - self.Q) / self.N)

    def __str__(self):
        return f"Node at {hex(id(self))} with Q={self.Q:.3f}, N={self.N}"

    def __repr__(self):
        return f"Node at {hex(id(self))} with Q={self.Q:.3f}, N={self.N}"
