from typing import Optional, Tuple

import numpy as np

from cdrl.agent.ordering.ordering_state import OrderingState
from cdrl.agent.ordering.ordering_local_score import (
    OrderingLocalScoreCache,
    local_score,
)
from cdrl.agent.ordering.ordering_decoder import decode_ordering
from cdrl.state.dag_state import DAGState


def _terminal_reward(
    X: np.ndarray,
    order: Tuple[int, ...],
    reg_type: str,
    bic_penalty: float,
    score_type: str,
    max_indegree: int,
    coeff_threshold: float,
    rfun,
    cache: Optional[OrderingLocalScoreCache] = None,
) -> Tuple[float, DAGState]:
    """
    Decode a complete (or partial) ordering to a pruned DAGState and score it.

    This is the single point where a rollout terminal is evaluated.  The pruned
    DAG is scored by rfun.calculate_reward_single_graph so MCTS simulations are
    guided by the exact same metric that final evaluation uses.

    Returns:
        (reward, dag) - reward is rfun(dag), dag is the decoded+pruned DAGState.
    """
    dag = decode_ordering(
        X, order, reg_type, bic_penalty, score_type,
        max_indegree, coeff_threshold, cache=cache,
        apply_pruning=False,
    )
    reward = float(rfun.calculate_reward_single_graph(dag, _skip_cycle_check=True))

    # Normalise partial orderings to the full d-node BIC scale by adding the
    # null-model local score (zero parents) for every unplaced variable.
    # This is equivalent to evaluating a full d-node DAG where unplaced nodes
    # are isolated, giving a consistent lower bound on the true full-ordering
    # reward so that UCT Q-values remain comparable across tree depths.
    d = X.shape[1]
    k = len(order)
    if k < d:
        placed_mask = np.zeros(d, dtype=bool)
        placed_mask[list(order)] = True
        if cache is not None and cache.null_scores is not None:
            reward += float(cache.null_scores[~placed_mask].sum())
        else:
            n = X.shape[0]
            for j in np.where(~placed_mask)[0]:
                resid = X[:, j] - X[:, j].mean()
                rss = float(resid @ resid)
                # Match ContinuousVarsBICRewardFunction: GPR adds a constant +1.0
                # to every node's RSS, including the null model.
                if reg_type == "GPR":
                    rss += 1.0
                reward += float(-np.log(rss / n + 1e-8))

    return reward, dag


def rollout_random(
    X: np.ndarray,
    state: OrderingState,
    rng: np.random.Generator,
    reg_type: str,
    bic_penalty: float,
    score_type: str,
    max_indegree: int,
    coeff_threshold: float,
    rfun,
    cache: Optional[OrderingLocalScoreCache] = None,
    rollout_depth: int = 0,
) -> Tuple[float, Tuple[int, ...], DAGState]:
    """
    Complete the partial ordering by uniform random variable selection,
    then decode and score the resulting DAG.

    rollout_depth == 0:  complete to full terminal (all d variables placed).
    rollout_depth > 0:   extend by exactly min(rollout_depth, remaining) steps,
                         then evaluate the (possibly partial) ordering.

    Returns:
        (reward, completed_order, dag)
    """
    order = list(state.order)
    remaining = list(state.available_actions())
    rng.shuffle(remaining)

    steps = len(remaining) if rollout_depth == 0 else min(rollout_depth, len(remaining))
    order.extend(remaining[:steps])

    completed_order = tuple(order)
    reward, dag = _terminal_reward(
        X, completed_order, reg_type, bic_penalty, score_type,
        max_indegree, coeff_threshold, rfun, cache=cache,
    )
    return reward, completed_order, dag


def rollout_greedy(
    X: np.ndarray,
    state: OrderingState,
    rng: np.random.Generator,
    reg_type: str,
    bic_penalty: float,
    score_type: str,
    max_indegree: int,
    coeff_threshold: float,
    rfun,
    cache: Optional[OrderingLocalScoreCache] = None,
    rollout_depth: int = 0,
) -> Tuple[float, Tuple[int, ...], DAGState]:
    """
    Complete the partial ordering by greedy variable selection: at each step,
    choose the variable v that maximises local_score(v, last_k_predecessors).

    The last min(max_indegree, len(current_order)) variables in the current prefix
    are used as candidate parents for scoring, matching the decoder's Stage 2 logic.
    Ties are broken randomly using rng.

    rollout_depth == 0: complete to full terminal.
    rollout_depth > 0:  extend by min(rollout_depth, remaining) greedy steps.

    Returns:
        (reward, completed_order, dag)
    """
    order = list(state.order)
    avail = set(state.available_actions())
    d = state.d

    steps = len(avail) if rollout_depth == 0 else min(rollout_depth, len(avail))

    for _ in range(steps):
        if not avail:
            break

        # Candidate parents for each variable = last k in the current prefix
        k = len(order) if max_indegree < 0 else min(max_indegree, len(order))
        parents_for_cand = order[-k:] if k > 0 else []

        best_sc = float("-inf")
        best_vars = []
        for v in avail:
            sc = local_score(
                X, v, parents_for_cand, reg_type, bic_penalty, score_type, d, cache=cache
            )
            if sc > best_sc:
                best_sc = sc
                best_vars = [v]
            elif sc == best_sc:
                best_vars.append(v)

        chosen = int(rng.choice(best_vars)) if len(best_vars) > 1 else best_vars[0]
        order.append(chosen)
        avail.discard(chosen)

    completed_order = tuple(order)
    reward, dag = _terminal_reward(
        X, completed_order, reg_type, bic_penalty, score_type,
        max_indegree, coeff_threshold, rfun, cache=cache,
    )
    return reward, completed_order, dag
