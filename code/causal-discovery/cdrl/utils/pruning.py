#!/usr/bin/env python

"""
Coefficient-based edge pruning for linear and polynomial causal regression models.

This script was integrated without modification from the RL-BIC repository.

Both functions fit a regression model for each node using its candidate parents
and discard any parent whose fitted coefficient falls below a threshold `th`.
`graph_pruned_by_coef` uses linear regression; `graph_pruned_by_coef_2nd` uses
quadratic (polynomial degree-2) regression.
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def graph_prunned_by_coef(graph_batch, X, th=0.3):
    """
    Prunes graph edges by linear regression coefficient magnitude.

    For each node i, fits a linear regression of i on its candidate parents
    (rows in graph_batch with nonzero entries) and removes any parent whose
    absolute fitted coefficient is below the threshold `th`. Returns a binary
    adjacency matrix of shape (d, d) as a float32 array.
    """
    d = len(graph_batch)
    reg = LinearRegression()
    W = []

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    return np.float32(np.abs(W) > th)


def graph_prunned_by_coef_2nd(graph_batch, X, th=0.3):
    """
    Prunes graph edges by quadratic regression coefficient magnitude.

    Like `graph_pruned_by_coef` but fits a degree-2 polynomial regression
    instead of a linear one. A parent is retained if any of its polynomial
    basis coefficients (linear or quadratic terms) exceeds the threshold `th`.
    Returns a list of per-node coefficient arrays of length d.
    """
    d = len(graph_batch)
    reg = LinearRegression()
    poly = PolynomialFeatures()
    W = []

    for i in range(d):
        col = graph_batch[i] > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        X_train_expand = poly.fit_transform(X_train)[:, 1:]
        X_train_expand_names = poly.get_feature_names()[1:]

        y = X[:, i]
        reg.fit(X_train_expand, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )

        for ci in range(d):
            if col[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):
                    if xxi in xxx:
                        if np.abs(reg_coeff[iii]) > th:
                            new_reg_coeff[ci] = 1.0
                            break
                cj += 1
        W.append(new_reg_coeff)

    return W
