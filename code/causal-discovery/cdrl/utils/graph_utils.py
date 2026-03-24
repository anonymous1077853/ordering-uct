"""
Graph utility functions: cycle detection, adjacency-matrix conversions, and networkx helpers.

Two cycle-detection approaches are provided:
- NOTEARS-based (`contains_cycles`): uses the matrix-exponential characterisation
  tr(exp(A)) - d = 0 iff A is acyclic. Fast but approximate (floating-point threshold).
- DFS-based (`contains_cycles_exact`): delegates to networkx, exact but slower.
"""

import numpy as np
from scipy.linalg import expm as matrix_exponential
import networkx as nx

# Tolerance for the NOTEARS acyclicity measure tr(exp(A)) - d.
# A graph is treated as acyclic when the measure falls below this value.
CYCNESS_THRESHOLD = 1e-5


def check_contains_undirected(adj_matrix):
    """Checks whether the adjacency matrix has any undirected edges."""
    eq_mask = (adj_matrix.T == adj_matrix)
    return (eq_mask & adj_matrix == 1).any()


def split_directed_undirected(adj_matrix):
    """Splits the adjacency matrix into directed (asymmetric) and undirected (symmetric) components."""
    n = adj_matrix.shape[0]
    eq_mask = (adj_matrix.T == adj_matrix)
    undirected_mask = (eq_mask & adj_matrix == 1)
    undirected_idx = np.nonzero(undirected_mask)
    directed_mask = (~eq_mask & adj_matrix == 1)
    directed_idx = np.nonzero(directed_mask)
    adj_directed = np.zeros((n, n), dtype=np.int32)
    adj_directed[directed_idx] = 1
    adj_undirected = np.zeros((n, n), dtype=np.int32)
    adj_undirected[undirected_idx] = 1
    adj_undirected = np.triu(adj_undirected)
    return adj_directed, adj_undirected


def get_int_representations(adj_matrix):
    """
    Returns two integer encodings of the adjacency matrix used by the BIC score cache.

    Each row of the matrix is read as a binary number; `graph_to_int` prepends a
    row-index offset (`2^d * i`) to make the encoding globally unique across rows,
    while `graph_to_int2` stores the raw row value. Both lists have length d.
    """
    maxlen = adj_matrix.shape[0]
    baseint = 2 ** maxlen

    graph_to_int = []
    graph_to_int2 = []
    for i in range(maxlen):
        adj_matrix[i][i] = 0
        tt = np.int32(adj_matrix[i])
        graph_to_int.append(baseint * i + np.int(''.join([str(ad) for ad in tt]), 2))
        graph_to_int2.append(np.int(''.join([str(ad) for ad in tt]), 2))
    return graph_to_int, graph_to_int2


def compute_cycness(adj_matrix):
    """
    Returns the NOTEARS acyclicity measure tr(exp(A)) - d for the given adjacency matrix.

    The measure equals zero exactly when the graph is acyclic and grows monotonically
    with the number and weight of cycles. Compare against CYCNESS_THRESHOLD to decide
    whether a graph should be treated as acyclic.
    """
    maxlen = adj_matrix.shape[0]
    cycness = np.trace(matrix_exponential(np.array(adj_matrix))) - maxlen
    return cycness


def contains_cycles(adj_matrix):
    """
    Returns True if the adjacency matrix contains a cycle, using the NOTEARS measure.

    This is an approximate test - floating-point rounding means very small cycles may
    fall below CYCNESS_THRESHOLD. Use `contains_cycles_exact` when correctness matters
    more than speed.
    """
    cycness = compute_cycness(adj_matrix)
    return cycness > CYCNESS_THRESHOLD


def contains_cycles_exact(nx_graph):
    """
    Returns True if the networkx DiGraph contains a cycle, using an exact DFS traversal.

    Slower than `contains_cycles` but correct for all graph structures.
    """
    return (not nx.algorithms.dag.is_directed_acyclic_graph(nx_graph))


def nx_graph_from_adj_matrix(adj_matrix):
    """Converts an adjacency matrix to a networkx DiGraph."""
    return nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)


def nx_graph_to_adj_matrix(nx_graph, nodelist=None):
    """Converts a networkx DiGraph to an integer adjacency matrix."""
    if nodelist is None:
        nodelist = np.arange(nx_graph.number_of_nodes())
    return np.asarray(nx.convert_matrix.to_numpy_array(nx_graph, nodelist=nodelist, dtype=np.int32))


def edge_list_from_adj_matrix(adj_matrix):
    """Returns the list of (src, dst) edge tuples corresponding to nonzero entries of the adjacency matrix."""
    nonzero_idx = np.transpose(np.nonzero(adj_matrix))
    edge_list = [tuple(row) for row in nonzero_idx]
    return edge_list


def edge_list_to_nx_graph(edge_list, num_nodes):
    """Constructs a networkx DiGraph from an edge list, ensuring all node indices up to num_nodes are present."""
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from(edge_list)
    return nx_graph
