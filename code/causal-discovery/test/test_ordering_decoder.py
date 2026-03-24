import numpy as np

from cdrl.agent.ordering.ordering_decoder import decode_ordering


def test_decode_ordering_max_indegree_minus_one_means_no_cap():
    # Simple 2-variable linear relation: X1 depends strongly on X0.
    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.normal(size=n)
    x1 = x0 + 0.01 * rng.normal(size=n)
    X = np.column_stack([x0, x1]).astype(np.float32)

    order = (0, 1)
    bic_penalty = float(np.log(n) / n)

    g = decode_ordering(
        X=X,
        order=order,
        reg_type="LR",
        bic_penalty=bic_penalty,
        score_type="BIC_different_var",
        max_indegree=-1,  # sentinel for "auto"
        coeff_threshold=0.3,
        cache=None,
        apply_pruning=False,
    )

    adj = g.get_adjacency_matrix()
    assert int(adj[0, 1]) == 1


def test_decode_ordering_max_indegree_zero_blocks_all_edges():
    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.normal(size=n)
    x1 = x0 + 0.01 * rng.normal(size=n)
    X = np.column_stack([x0, x1]).astype(np.float32)

    order = (0, 1)
    bic_penalty = float(np.log(n) / n)

    g = decode_ordering(
        X=X,
        order=order,
        reg_type="LR",
        bic_penalty=bic_penalty,
        score_type="BIC_different_var",
        max_indegree=0,
        coeff_threshold=0.3,
        cache=None,
        apply_pruning=False,
    )

    adj = g.get_adjacency_matrix()
    assert int(adj.sum()) == 0

