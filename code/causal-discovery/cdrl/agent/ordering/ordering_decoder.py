from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from cdrl.state.dag_state import DAGState
from cdrl.agent.ordering.ordering_local_score import (
    OrderingLocalScoreCache,
    best_candidate_gram,
    local_score,
)



def _decode_stage2_one(
    v: int,
    candidates: List[int],
    cache: Optional[OrderingLocalScoreCache],
    X: np.ndarray,
    reg_type: str,
    bic_penalty: float,
    max_indegree: int,
    d: int,
    stage2_score_type: str,
) -> List[int]:
    """
    Run Stage 2 (forward greedy parent selection + backward sweep) for one variable.

    This is the inner body of the Stage 2 loop in decode_ordering(), extracted so
    that _run_search() can call it to precompute prefix parent sets once per step.

    Args:
        v:                Target variable index.
        candidates:       Ordering-consistent candidate parents (all predecessors of v).
        cache:            Shared local-score cache; may be None.
        X:                Data matrix (n, d).
        reg_type:         "LR", "QR", or "GPR".
        bic_penalty:      log(n)/n.
        max_indegree:     Maximum number of parents to select.
        d:                Number of variables (for BIC penalty scaling).
        stage2_score_type: "BIC_different_var" (always passed as such from decode_ordering).

    Returns:
        Selected parent list for v (subset of candidates).
    """
    if not candidates:
        return []

    # max_indegree == -1 is used across the codebase as a sentinel for "auto"/no cap.
    # In that case, allow up to all candidates.
    if max_indegree < 0:
        max_indegree = len(candidates)

    _use_gram = cache is not None and cache.G_aug is not None and reg_type == "LR"
    max_parents = min(max_indegree, len(candidates))

    # Exact optimum via full subset enumeration when the search space is small.
    # For Sachs (max_indegree=3, <=10 candidates): C(10,0)+...+C(10,3)=176 subsets per node.
    # Avoids the suboptimality of greedy forward selection at negligible extra cost.
    _ENUM_INDEGREE_THRESHOLD = 4
    _ENUM_CANDIDATES_THRESHOLD = 15
    if max_parents <= _ENUM_INDEGREE_THRESHOLD and len(candidates) <= _ENUM_CANDIDATES_THRESHOLD:
        best_sc = local_score(X, v, [], reg_type, bic_penalty, stage2_score_type, d, cache=cache)
        best_subset: List[int] = []
        for size in range(1, max_parents + 1):
            for subset in combinations(candidates, size):
                sc = local_score(
                    X, v, list(subset), reg_type, bic_penalty, stage2_score_type, d, cache=cache
                )
                if sc > best_sc:
                    best_sc = sc
                    best_subset = list(subset)
        return best_subset

    selected: List[int] = []
    remaining = set(candidates)

    if _use_gram:
        _G = cache.G_aug
        _n = cache.n
        _d_bias = cache.d

        Pa_aug = [_d_bias]                                  # bias only (empty Pa)
        XtX_Pa = _G[np.ix_(Pa_aug, Pa_aug)]                # [[n]]
        Xty_Pa = _G[Pa_aug, v]                              # [sum(X[:,v])]
        theta = np.linalg.solve(XtX_Pa, Xty_Pa)
        rss_Pa = float(_G[v, v] - theta @ Xty_Pa)          # RSS with no parents
        base_score = float(-np.log(rss_Pa / _n + 1e-8))    # k=0, no penalty

        for _ in range(max_parents):
            if not remaining:
                break
            best_u, best_sc, best_rss = best_candidate_gram(
                _G, _n, v, Pa_aug, XtX_Pa, Xty_Pa, rss_Pa,
                remaining, base_score, bic_penalty,
            )
            if best_u is None:
                break
            selected.append(best_u)
            remaining.discard(best_u)
            base_score = best_sc
            rss_Pa = best_rss
            # Rebuild Pa_aug and normal-equations matrices for next iteration.
            Pa_aug = selected + [_d_bias]
            XtX_Pa = _G[np.ix_(Pa_aug, Pa_aug)]
            Xty_Pa = _G[Pa_aug, v]
    else:
        # Original path for QR / GPR or when Gram cache is absent.
        base_score = local_score(
            X, v, [], reg_type, bic_penalty, stage2_score_type, d, cache=cache
        )
        for _ in range(max_parents):
            if not remaining:
                break
            best_sc = base_score
            best_add = None
            for u in remaining:
                sc = local_score(
                    X, v, selected + [u], reg_type, bic_penalty, stage2_score_type, d, cache=cache
                )
                if sc > best_sc:
                    best_sc = sc
                    best_add = u
            if best_add is None:
                break  # no addition improves score; stop early
            selected.append(best_add)
            remaining.discard(best_add)
            base_score = best_sc

    # Backward sweep: remove any selected parent whose removal improves the score.
    # Ties (sc >= base_score) favour fewer parents (Occam's razor).
    # Restart after each removal to handle cascading effects; converges in O(k) steps.
    improved = True
    while improved:
        improved = False
        for u in list(selected):
            without_u = [x for x in selected if x != u]
            sc = local_score(
                X, v, without_u, reg_type, bic_penalty, stage2_score_type, d, cache=cache
            )
            if sc >= base_score:
                selected = without_u
                base_score = sc
                improved = True
                break  # restart sweep after each removal

    return selected


def decode_ordering(
    X: np.ndarray,
    order: Tuple[int, ...],
    reg_type: str,
    bic_penalty: float,
    score_type: str,
    max_indegree: int,
    coeff_threshold: float,
    cache: Optional[OrderingLocalScoreCache] = None,
    apply_pruning: bool = True,
) -> DAGState:
    """
    Convert a complete variable ordering into a DAGState via 2- or 3-stage decoding.

    Stage 1 - Ordering-consistent supergraph:
        Each variable v gets all its predecessors in the ordering as candidate parents.

    Stage 2 - Forward greedy parent selection (max_indegree constraint):
        For each node v, iteratively add the parent from the candidate set that most
        improves local_score(v, current_Pa).  Stop when max_indegree is reached or
        no further addition improves the score.

        If cache.prefix_parent_sets contains an entry for v (set by _run_search()
        when v was committed to the prefix), Stage 2 is skipped for v and the cached
        result is used directly.  This avoids redundant computation across simulations
        that share the same committed prefix.

    Stage 3 - Statistical pruning (only when apply_pruning=True):
        LR: graph_prunned_by_coef   (OLS coefficient threshold)
        QR: graph_prunned_by_coef_2nd (polynomial coefficient threshold)
        GPR: pruning_cam             (R-based CAM / GAM p-value threshold)

    The returned DAGState is guaranteed acyclic (edges only point forward in the
    ordering) and is ready for rfun.calculate_reward_single_graph().

    Args:
        X:               Data matrix (n, d), already normalised.
        order:           Complete ordering - a permutation of {0, ..., d-1}.
        reg_type:        "LR", "QR", or "GPR".
        bic_penalty:     log(n)/n - taken directly from rfun.bic_penalty.
        score_type:      "BIC_different_var" or "BIC".
        max_indegree:    Maximum parent-set size enforced in Stage 2.
        coeff_threshold: |beta| cutoff for LR/QR Stage 3 pruning (default 0.3).
        cache:           Optional shared RSS cache for Stage 2 local scoring.
        apply_pruning:   Whether to apply Stage 3 statistical pruning.
                         Set False during MCTS simulations for speed; set True
                         only for the final decode used for metric reporting.

    Returns:
        DAGState built from the adjacency matrix after Stages 1+2 (and optionally 3).
    """
    n, d = X.shape

    # Stage 1: ordering-consistent supergraph
    # parent_sets[v] = all predecessors of v (in ordering order)
    parent_sets: List[List[int]] = [[] for _ in range(d)]
    for pos, v in enumerate(order):
        parent_sets[v] = list(order[:pos])

    # Stage 2: forward greedy parent selection
    # Always use BIC_different_var for the decomposable per-node scoring criterion,
    # regardless of the global score_type.  BIC_different_var is the only variant
    # whose per-node penalty (k * bic_penalty) exactly decomposes the global reward;
    # BIC uses an approximation (k * bic_penalty / d) that can mislead greedy selection.
    # The global score_type is used only by rfun.calculate_reward_single_graph.
    stage2_score_type = "BIC_different_var"

    for v in order:
        # Prefix cache: if _run_search() pre-decoded this variable for the current
        # committed prefix, reuse the cached result and skip Stage 2 entirely.
        if cache is not None and v in cache.prefix_parent_sets:
            parent_sets[v] = cache.prefix_parent_sets[v]
            continue

        parent_sets[v] = _decode_stage2_one(
            v, parent_sets[v], cache, X, reg_type, bic_penalty,
            max_indegree, d, stage2_score_type,
        )

    # Build adjacency matrix from parent_sets (i->j format)
    adj_ij = np.zeros((d, d), dtype=np.float32)
    for v in range(d):
        for u in parent_sets[v]:
            adj_ij[u, v] = 1.0   # edge u -> v

    # Stage 3: statistical pruning (final decode only)
    if apply_pruning:
        adj_ji = adj_ij.T  # j->i format expected by graph_prunned_by_coef
        adj_final = _apply_stage3_pruning(adj_ij, adj_ji, X, reg_type, coeff_threshold)
    else:
        adj_final = adj_ij.astype(np.int32)

    # Build DAGState
    nx_graph = nx.from_numpy_array(adj_final, create_using=nx.DiGraph)
    return DAGState(nx_graph, init_tracking=False)


def _apply_stage3_pruning(
    adj_ij: np.ndarray,
    adj_ji: np.ndarray,
    X: np.ndarray,
    reg_type: str,
    coeff_threshold: float,
) -> np.ndarray:
    """
    Apply Stage 3 pruning using the existing cdrl utility functions.

    Pruning convention notes (matching get_metrics_for_graph in eval_utils.py):
      - graph_prunned_by_coef / graph_prunned_by_coef_2nd:
          take adj_ji (j->i), return adj_ji -> caller transposes back to i->j
      - pruning_cam:
          takes adj_ij (i->j), returns adj_ij (no transpose needed)

    Args:
        adj_ij: Adjacency matrix in i->j format (shape dxd, float32).
        adj_ji: Adjacency matrix in j->i format (transposed view of adj_ij).
        X:      Data matrix (n, d).
        reg_type: "LR", "QR", or "GPR".
        coeff_threshold: |beta| cutoff for LR/QR pruning.

    Returns:
        Pruned adjacency matrix in i->j format (dtype int32).
    """
    if reg_type == "LR":
        from cdrl.utils.pruning import graph_prunned_by_coef
        pruned_ji = np.array(
            graph_prunned_by_coef(adj_ji, X, th=coeff_threshold),
            dtype=np.int32,
        )
        return pruned_ji.T  # convert back to i->j

    elif reg_type == "QR":
        from cdrl.utils.pruning import graph_prunned_by_coef_2nd
        pruned_ji = np.array(
            graph_prunned_by_coef_2nd(adj_ji, X, th=coeff_threshold),
            dtype=np.int32,
        )
        return pruned_ji.T  # convert back to i->j

    elif reg_type == "GPR":
        from cdrl.utils.cam_with_pruning_cam import pruning_cam
        pruned_ij = pruning_cam(X, adj_ij)
        return np.array(pruned_ij, dtype=np.int32)

    else:
        raise ValueError(f"Unknown reg_type: {reg_type!r}")
