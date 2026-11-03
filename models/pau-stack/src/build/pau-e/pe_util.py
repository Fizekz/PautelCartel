"""pau-e small helpers: OLS with cluster-robust SEs (numpy only)."""
import numpy as np


def ols(y, X, groups=None):
    """Returns (beta, se, t). X without constant; a constant is added first. Cluster-robust if groups given."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ X.T @ y
    e = y - X @ b
    if groups is None:
        V = XtX * (e @ e) / max(len(y) - X.shape[1], 1)
    else:
        g = np.asarray(groups)
        S = np.zeros((X.shape[1], X.shape[1]))
        for k in np.unique(g):
            s = X[g == k].T @ e[g == k]
            S += np.outer(s, s)
        G = len(np.unique(g))
        V = XtX @ S @ XtX * G / max(G - 1, 1)
    se = np.sqrt(np.diag(V))
    return b, se, b / se
