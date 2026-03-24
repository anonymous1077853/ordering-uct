from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import numpy as np


class OrderingLocalScoreCache:
    """
    Cache for RSS computations keyed by (node_index, frozenset(parent_indices)).
    A single cache instance is shared across all MCTS simulations within one
    eval() call, providing major speedups when the same parent sets recur.

    When precompute(X) is called (LR only), the augmented Gram matrix
    G_aug = [X|ones].T @ [X|ones] is stored.  This lets _compute_rss avoid
    the O(n*k^2) matrix-multiply on cache misses, replacing it with an O(k^2)
    submatrix slice plus an O(k^3) solve.
    """

    def __init__(self) -> None:
        self._rss: Dict[Tuple[int, FrozenSet[int]], float] = {}
        self.G_aug: Optional[np.ndarray] = None   # set by precompute()
        self.null_scores: Optional[np.ndarray] = None  # shape (d,), set by precompute()
        self.n: int = 0
        self.d: int = 0   # bias column index in G_aug = d (number of variables)
        # Maps variable index -> its Stage-2 selected parent list for the current
        # committed prefix.  Populated incrementally by _run_search() as variables
        # are committed; lets decode_ordering() skip Stage 2 for prefix variables.
        self.prefix_parent_sets: Dict[int, List[int]] = {}

    def precompute(self, X: np.ndarray) -> None:
        """
        Precompute the augmented Gram matrix G_aug for fast LR regression.

        G_aug = X_aug.T @ X_aug  where  X_aug = [X | ones(n,1)]  (shape n x d+1).

        After this call:
          G_aug[i, j]  (i,j < d)  =  X[:,i] @ X[:,j]
          G_aug[d, j]  (j < d)    =  sum(X[:,j])        (bias x column j)
          G_aug[d, d]              =  n
          G_aug[j, j]              =  X[:,j] @ X[:,j]   (yTy for target j)

        Call once per eval() for LR instances; costs O(n*d^2) and pays back
        immediately on the first cache miss.
        """
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])   # (n, d+1)
        self.G_aug = X_aug.T @ X_aug   # (d+1, d+1)
        self.n = n
        self.d = d   # bias column index
        # Null-model local scores: local_score(j, Pa={}) = -log(RSS_j({})/n + 1e-8)
        # RSS_j({}) = ||X_j - mean(X_j)||^2 = G_aug[j,j] - G_aug[d,j]^2 / n
        rss_null = np.diag(self.G_aug)[:d] - self.G_aug[d, :d] ** 2 / n
        self.null_scores = -np.log(rss_null / n + 1e-8)

    def get(self, j: int, parents: FrozenSet[int]) -> Optional[float]:
        return self._rss.get((j, parents))

    def set(self, j: int, parents: FrozenSet[int], rss: float) -> None:
        self._rss[(j, parents)] = float(rss)


def _compute_rss(
    X: np.ndarray,
    j: int,
    parents: List[int],
    reg_type: str,
) -> float:
    """
    Compute RSS for predicting X[:, j] from X[:, parents].

    Replicates the regression logic of ContinuousVarsBICRewardFunction exactly:
      - LR:  X_aug = [X_pa | ones], OLS via np.linalg.solve
      - QR:  polynomial features on X_pa, then LR (matching calculate_QR)
      - GPR: GaussianProcessRegressor with median bandwidth + offset 1.0 (matching calculate_GPR)
    """
    y = X[:, j]
    n = X.shape[0]

    if len(parents) == 0:
        resid = y - np.mean(y)
        rss = float(resid @ resid)
        # Match ContinuousVarsBICRewardFunction: GPR shifts all RSS values by +1.0
        # (including the null model) to avoid log(~0) dominating the score.
        if reg_type == "GPR":
            rss += 1.0
        return rss

    X_pa = X[:, parents]

    if reg_type == "LR":
        ones = np.ones((n, 1), dtype=X_pa.dtype)
        X_aug = np.hstack([X_pa, ones])
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y
        theta = np.linalg.solve(XtX, Xty)
        resid = X_aug @ theta - y
        return float(resid @ resid)

    elif reg_type == "QR":
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures()
        X_poly = poly.fit_transform(X_pa)[:, 1:]  # drop bias column (added manually)
        ones = np.ones((n, 1), dtype=X_poly.dtype)
        X_aug = np.hstack([X_poly, ones])
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y
        theta = np.linalg.solve(XtX, Xty)
        resid = X_aug @ theta - y
        return float(resid @ resid)

    elif reg_type == "GPR":
        from scipy.spatial.distance import pdist
        from sklearn.gaussian_process import GaussianProcessRegressor as GPR
        med_w = np.median(pdist(X_pa, "euclidean"))
        if med_w <= 0:
            med_w = 1.0
        gpr = GPR().fit(X_pa / med_w, y)
        yhat = gpr.predict(X_pa / med_w)
        resid = y - yhat
        rss = float(resid @ resid) + 1.0  # +1.0 offset matches cdrl convention
        return rss

    else:
        raise ValueError(f"Unknown reg_type: {reg_type!r}")


def local_rss(
    X: np.ndarray,
    j: int,
    parents: Iterable[int],
    reg_type: str,
    cache: Optional[OrderingLocalScoreCache] = None,
) -> float:
    """Cached RSS computation for node j given its parent set."""
    pset = frozenset(parents)   # parents are already ints; no int() conversion needed
    if cache is not None:
        hit = cache.get(j, pset)
        if hit is not None:
            return hit

    sorted_parents = sorted(pset)

    if reg_type == "LR" and cache is not None and cache.G_aug is not None:
        # Fast path: use precomputed Gram matrix.  Avoids the O(n*k^2) matrix
        # multiply and residual computation; replaces it with O(k^2) slicing +
        # O(k^3) solve + O(k) dot product.
        G = cache.G_aug
        d_bias = cache.d            # index of the bias column in G_aug
        idx = sorted_parents + [d_bias]   # augmented parent indices
        XtX = G[np.ix_(idx, idx)]         # (k+1, k+1)
        Xty = G[idx, j]                   # (k+1,)
        theta = np.linalg.solve(XtX, Xty)
        rss = float(G[j, j] - theta @ Xty)   # RSS = yTy - theta.TXty
    else:
        rss = _compute_rss(X, j, sorted_parents, reg_type)

    if cache is not None:
        cache.set(j, pset, rss)
    return rss


def local_score(
    X: np.ndarray,
    j: int,
    parents: Iterable[int],
    reg_type: str,
    bic_penalty: float,
    score_type: str,
    d: int,
    cache: Optional[OrderingLocalScoreCache] = None,
) -> float:
    """
    Per-node decomposable BIC-like score to MAXIMIZE for node j given parents.

    Aligned with ContinuousVarsBICRewardFunction.bic_from_RSS_list:

    BIC_different_var (decomposable, exact per-node contribution):
        S_j = -log(RSS_j/n + 1e-8) - |Pa_j| * bic_penalty
        where bic_penalty = log(n)/n  (= rfun.bic_penalty)

    BIC (approximation - not decomposable globally, but used here for greedy
         parent selection guidance only; exact reward comes from rfun):
        S_j = -log(RSS_j/n + 1e-8) - |Pa_j| * bic_penalty / d

    Args:
        X:           Data matrix (n, d).
        j:           Node index to score.
        parents:     Parent indices for node j.
        reg_type:    "LR", "QR", or "GPR".
        bic_penalty: log(n)/n - taken directly from rfun.bic_penalty.
        score_type:  "BIC_different_var" or "BIC".
        d:           Number of variables (used for BIC penalty scaling).
        cache:       Optional shared RSS cache.

    Returns:
        Local score (higher is better).
    """
    n = X.shape[0]
    pset = frozenset(parents)   # parents are already ints
    k = len(pset)
    rss = local_rss(X, j, pset, reg_type, cache=cache)

    fit_term = np.log(rss / n + 1e-8)

    if score_type == "BIC_different_var":
        penalty = k * bic_penalty
    else:  # BIC - per-node approximation
        penalty = k * bic_penalty / d

    return float(-(fit_term + penalty))


def best_candidate_gram(
    G_aug: np.ndarray,
    n: int,
    j: int,
    Pa_aug: List[int],
    XtX_Pa: np.ndarray,
    Xty_Pa: np.ndarray,
    rss_Pa: float,
    candidates: Iterable[int],
    base_score: float,
    bic_penalty: float,
) -> Tuple[Optional[int], float, float]:
    """
    Vectorised forward-selection step for LR using the precomputed Gram matrix.

    Evaluates adding each variable in `candidates` to the current parent set Pa
    simultaneously via the Schur complement (rank-1 RSS update), then returns
    the candidate that yields the highest local score.

    The Schur complement formula gives, for each candidate u:
        RSS(Pa | {u}) = RSS(Pa) - (cross_y_u)^2 / S_uu
    where:
        S_uu     = effective variance of u after projecting out Pa
        cross_y  = effective covariance of u with target j after projecting out Pa

    Both quantities are derived from G_aug without touching X, so the dominant
    cost is a single batched solve of size (|Pa|+1, |U|) instead of |U| separate
    (|Pa|+2)-sized solves.

    Args:
        G_aug:      Precomputed augmented Gram matrix (d+1, d+1); last index is bias.
        n:          Number of data points.
        j:          Target variable index.
        Pa_aug:     Augmented indices for current parent set: list(Pa) + [d_bias].
        XtX_Pa:     Normal-equations matrix for Pa_aug, shape (|Pa_aug|, |Pa_aug|).
        Xty_Pa:     Normal-equations RHS for Pa_aug, shape (|Pa_aug|,).
        rss_Pa:     Current RSS for target j given Pa.
        candidates: Candidate parent variable indices to evaluate.
        base_score: Current local score (for Pa); must be beaten to accept a candidate.
        bic_penalty: log(n)/n, matching BIC_different_var penalty.

    Returns:
        (best_u, best_score, best_rss):
            best_u      - index of the best candidate, or None if none improves.
            best_score  - local score after adding best_u (= base_score if None).
            best_rss    - RSS after adding best_u (= rss_Pa if None).
    """
    U = list(candidates)
    if not U:
        return None, base_score, rss_Pa

    # cross[:, i] = G_aug[Pa_aug, U[i]]  - cross-products of each Pa member with U[i]
    cross = G_aug[np.ix_(Pa_aug, U)]                    # (|Pa_aug|, |U|)
    # One batched solve for all candidates simultaneously
    coeff = np.linalg.solve(XtX_Pa, cross)              # (|Pa_aug|, |U|)

    # Effective variance of each candidate after projecting out Pa
    S_uu = G_aug.diagonal()[U] - np.einsum('ij,ij->j', cross, coeff)  # (|U|,)

    # Effective covariance of each candidate with target j after projecting out Pa
    cross_y = G_aug[U, j] - coeff.T @ Xty_Pa           # (|U|,)

    # RSS when adding each candidate: RSS(Pa | {u}) = RSS(Pa) - cross_y^2 / S_uu
    rss_new = rss_Pa - cross_y ** 2 / np.maximum(S_uu, 1e-12)   # (|U|,)

    # Local score = -log(RSS/n + 1e-8) - k_new * bic_penalty
    # k_new = number of parents after adding u = len(Pa_aug) [bias already counted in Pa_aug]
    k_new = len(Pa_aug)
    score_new = -np.log(np.maximum(rss_new, 0.0) / n + 1e-8) - k_new * bic_penalty  # (|U|,)

    best_idx = int(np.argmax(score_new))
    if score_new[best_idx] > base_score:
        return U[best_idx], float(score_new[best_idx]), float(rss_new[best_idx])
    return None, base_score, rss_Pa
