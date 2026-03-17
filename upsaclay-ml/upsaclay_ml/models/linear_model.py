# upsaclay-ml/upsaclay_ml/models/linear_models.py

from __future__ import annotations
from typing import List, Sequence
from .estimator import Supervised

Number = float | int


def _as_point(x) -> List[float]:
    """Convert a sample to a numeric vector.
    Supports X[i] being a vector or a (vector, y) tuple.
    """
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    return [float(v) for v in x]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


class LinearRegression(Supervised):
    """
    Simple Linear Regression trained with batch Gradient Descent.

    Parameters
    ----------
    lr : float
        Learning rate.
    n_iter : int
        Number of gradient descent iterations.
    fit_intercept : bool
        Whether to learn an intercept term.
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000, fit_intercept: bool = True):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.fit_intercept = bool(fit_intercept)

        # learned params
        self.coef_ = None        # List[float]
        self.intercept_ = None   # float

    def fit(self, X, y=None):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        n = len(X)
        if n == 0:
            raise ValueError("X cannot be empty")
        if len(y) != n:
            raise ValueError("X and y must have the same length")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.n_iter <= 0:
            raise ValueError("n_iter must be a positive integer")

        X_list = [_as_point(X[i]) for i in range(n)]
        y_list = [float(v) for v in y]

        d = len(X_list[0])
        if d == 0:
            raise ValueError("Samples must have at least 1 feature")
        for i, p in enumerate(X_list):
            if len(p) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(p)}")

        # init params
        w = [0.0] * d
        b = 0.0

        # gradient descent
        for _ in range(self.n_iter):
            # predictions and errors
            # y_hat = Xw + b
            grad_w = [0.0] * d
            grad_b = 0.0

            for i in range(n):
                y_hat = _dot(X_list[i], w) + (b if self.fit_intercept else 0.0)
                err = y_hat - y_list[i]

                for j in range(d):
                    grad_w[j] += err * X_list[i][j]
                if self.fit_intercept:
                    grad_b += err

            # average gradients (MSE derivative uses 2/n factor; we can absorb 2 into lr)
            inv_n = 1.0 / n
            for j in range(d):
                w[j] -= self.lr * (grad_w[j] * inv_n)
            if self.fit_intercept:
                b -= self.lr * (grad_b * inv_n)

        self.coef_ = w
        self.intercept_ = b if self.fit_intercept else 0.0
        return self

    def predict(self, X, y=None):
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("This LinearRegression instance is not fitted yet. Call fit(X, y) first.")

        n = len(X)
        if n == 0:
            return []

        X_list = [_as_point(X[i]) for i in range(n)]
        d = len(self.coef_)
        for i, p in enumerate(X_list):
            if len(p) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(p)}")

        preds = []
        for i in range(n):
            preds.append(_dot(X_list[i], self.coef_) + self.intercept_)
        return preds
