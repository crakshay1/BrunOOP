# upsaclay_ml/models/tree.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, List, Any
import math

from .estimator import Supervised


def _as_point(x) -> List[float]:
    """Convert a sample to a numeric vector.
    Supports X[i] being a vector or a (vector, y) tuple.
    """
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    return [float(v) for v in x]


def _majority_class(y: Sequence[int]) -> int:
    counts = {}
    for c in y:
        counts[c] = counts.get(c, 0) + 1
    # deterministic tie-break: smallest label wins
    best = None
    for c in sorted(counts):
        if best is None or counts[c] > counts[best]:
            best = c
    return int(best)


def _entropy(y: Sequence[int]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = {}
    for c in y:
        counts[c] = counts.get(c, 0) + 1
    h = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def _gini(y: Sequence[int]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    counts = {}
    for c in y:
        counts[c] = counts.get(c, 0) + 1
    s = 0.0
    for cnt in counts.values():
        p = cnt / n
        s += p * (1.0 - p)
    return s


def _impurity(y: Sequence[int], criterion: str) -> float:
    if criterion == "entropy":
        return _entropy(y)
    if criterion == "gini":
        return _gini(y)
    raise ValueError("criterion must be 'gini' or 'entropy'")


@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    label: Optional[int] = None  # leaf label

    @property
    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTreeClassifier(Supervised):
    """
    Basic decision tree classifier for numeric features.
    Split test: x[feature] <= threshold -> left else right

    criterion: 'gini' or 'entropy' (impurity measures)
    """

    def __init__(self, max_depth: int = 10, criterion: str = "gini", \
                min_samples_split: int = 2):
        self.max_depth = int(max_depth)
        self.criterion = str(criterion)
        self.min_samples_split = int(min_samples_split)

        # learned attributes
        self.root_ = None
        self.n_features_in_ = None
        self.classes_ = None

    def fit(self, X, y=None):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        n = len(X)
        if n == 0:
            raise ValueError("X cannot be empty")
        if len(y) != n:
            raise ValueError("X and y must have the same length")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")

        X_list = [_as_point(X[i]) for i in range(n)]
        d = len(X_list[0])
        if d == 0:
            raise ValueError("Samples must have at least 1 feature")
        for i, row in enumerate(X_list):
            if len(row) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(row)}")

        y_list = [int(v) for v in y]
        self.n_features_in_ = d
        self.classes_ = sorted(set(y_list))

        self.root_ = self._build_tree(X_list, y_list, depth=0)
        return self

    def predict(self, X, y=None):
        if self.root_ is None:
            raise AttributeError("DecisionTreeClassifier is not fitted yet (root_ is None).")

        X_list = [_as_point(X[i]) for i in range(len(X))]
        preds = []
        for row in X_list:
            if self.n_features_in_ is not None and len(row) != self.n_features_in_:
                raise ValueError(f"Expected {self.n_features_in_} features, got {len(row)}")
            preds.append(self._predict_one(row, self.root_))
        return preds

    # ---------------- internals ----------------
    def _predict_one(self, x: List[float], node: Node) -> int:
        while not node.is_leaf:
            j = node.feature
            t = node.threshold
            # defensive
            if j is None or t is None:
                break
            node = node.left if x[j] <= t else node.right
            if node is None:
                break
        # should be leaf
        if node is None or node.label is None:
            # fallback (shouldn’t happen if built correctly)
            return self.classes_[0] if self.classes_ else 0
        return int(node.label)

    def _build_tree(self, X: List[List[float]], y: List[int], depth: int) -> Node:
        # stopping conditions (overfitting control mentioned in slides)
        # - pure node
        if len(set(y)) == 1:
            return Node(label=y[0])
        # - max depth
        if depth >= self.max_depth:
            return Node(label=_majority_class(y))
        # - too few samples
        if len(y) < self.min_samples_split:
            return Node(label=_majority_class(y))

        best = self._best_split(X, y)
        if best is None:
            return Node(label=_majority_class(y))

        feat, thr, left_idx, right_idx = best
        if not left_idx or not right_idx:
            return Node(label=_majority_class(y))

        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        node = Node(feature=feat, threshold=thr)
        node.left = self._build_tree(left_X, left_y, depth + 1)
        node.right = self._build_tree(right_X, right_y, depth + 1)
        return node

    def _best_split(self, X: List[List[float]], y: List[int]):
        n = len(y)
        d = len(X[0])

        parent_imp = _impurity(y, self.criterion)

        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_left = None
        best_right = None

        # For each feature, try thresholds between sorted unique values
        for j in range(d):
            vals = [X[i][j] for i in range(n)]
            uniq = sorted(set(vals))
            if len(uniq) <= 1:
                continue

            # candidate thresholds as midpoints
            thresholds = [(uniq[k] + uniq[k + 1]) / 2.0 for k in range(len(uniq) - 1)]
            for thr in thresholds:
                left_idx = [i for i in range(n) if X[i][j] <= thr]
                right_idx = [i for i in range(n) if X[i][j] > thr]
                if not left_idx or not right_idx:
                    continue

                left_y = [y[i] for i in left_idx]
                right_y = [y[i] for i in right_idx]

                left_imp = _impurity(left_y, self.criterion)
                right_imp = _impurity(right_y, self.criterion)

                # impurity reduction (Δ) described in the slides
                w_left = len(left_idx) / n
                w_right = len(right_idx) / n
                gain = parent_imp - (w_left * left_imp + w_right * right_imp)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thr = thr
                    best_left = left_idx
                    best_right = right_idx

        if best_feat is None or best_gain <= 0.0:
            return None
        return best_feat, best_thr, best_left, best_right
