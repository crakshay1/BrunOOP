# upsaclay-ml/upsaclay_ml/models/cluster.py

# imports
from __future__ import annotations
from .estimator import Unsupervised

from typing import List, Sequence, Optional
import random
import math


Number = float | int
Vector = Sequence[Number]


# ---- utility functions ----
def _as_point(x) -> List[float]:
    """Convert a sample to a numeric vector.
    Supports X[i] being a vector or a (vector, y) tuple.
    """
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    return [float(v) for v in x]


def _manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(ai - bi) for ai, bi in zip(a, b))


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

def _sqeuclidean(a: Sequence[float], b: Sequence[float]) -> float:
    # Squared Euclidean (no sqrt needed for comparisons)
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))

def _get_distance(name: str):
    if name == "manhattan":
        return _manhattan
    elif name == "euclidean":
        return _euclidean
    elif name == "sqeuclidean":
        return _sqeuclidean
    else:
        raise ValueError("metric must be 'manhattan' or 'euclidean' or 'sqeuclidean'")


def _mean(points: List[List[float]]) -> List[float]:
    d = len(points[0])
    out = [0.0] * d
    for p in points:
        for j in range(d):
            out[j] += p[j]
    n = float(len(points))
    return [v / n for v in out]


def _pairwise_dist_matrix(X_list: List[List[float]], dist_fn) -> List[List[float]]:
    """Compute a symmetric point-to-point distance matrix."""
    n = len(X_list)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_fn(X_list[i], X_list[j])
            D[i][j] = d
            D[j][i] = d
    return D



#---- K-Means Clustering ----
class KMeans(Unsupervised):
    """K-Means clustering."""

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        max_iter: int = 300,
        metric: str = "manhattan",
    ):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.max_iter = int(max_iter)
        self.metric = metric

        # learned attributes (ignored by Estimator.__repr__/__eq__ because they end with "_")
        self.labels_ = None
        self.centroids_ = None

    def fit(self, X, y=None):
        # ---- basic validation / normalization ----
        if X is None:
            raise ValueError("X cannot be None")
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("X cannot be empty")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than number of samples")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

        X_list = [_as_point(X[i]) for i in range(n_samples)]
        d = len(X_list[0])
        if d == 0:
            raise ValueError("Samples must have at least 1 feature")
        for i, p in enumerate(X_list):
            if len(p) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(p)}")

        rng = random.Random(self.random_state)

        # ---- initialize centroids: choose k distinct samples ----
        init_indices = rng.sample(range(n_samples), self.n_clusters)
        centroids = [X_list[idx][:] for idx in init_indices]

        labels = [-1] * n_samples
        old_labels = None

        # ---- main loop ----
        dist_fn = _get_distance(self.metric)
        for _ in range(self.max_iter):
            # assignment step
            for i, p in enumerate(X_list):
                best_j = 0
                best_dist = dist_fn(p, centroids[0])
                for j in range(1, self.n_clusters):
                    dist = dist_fn(p, centroids[j])
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                labels[i] = best_j

            # convergence check (labels unchanged)
            if old_labels is not None and labels == old_labels:
                break
            old_labels = labels[:]

            # update step
            clusters: List[List[List[float]]] = [[] for _ in range(self.n_clusters)]
            for i, lab in enumerate(labels):
                clusters[lab].append(X_list[i])

            for j in range(self.n_clusters):
                if clusters[j]:
                    centroids[j] = _mean(clusters[j])
                else:
                    # empty cluster: reinitialize centroid to a random data point
                    centroids[j] = X_list[rng.randrange(n_samples)][:]

        self.labels_ = labels
        self.centroids_ = centroids
        return self





# ---- Agglomerative Clustering ----
def _cluster_distance(A: List[int], B: List[int], D: List[List[float]], linkage: str) -> float:
    """Distance between two clusters A and B based on linkage, using point distance matrix D."""
    if linkage == "single":
        best = None
        for i in A:
            for j in B:
                d = D[i][j]
                if best is None or d < best:
                    best = d
        return float(best)

    if linkage == "complete":
        best = None
        for i in A:
            for j in B:
                d = D[i][j]
                if best is None or d > best:
                    best = d
        return float(best)

    if linkage == "average":
        total = 0.0
        count = 0
        for i in A:
            for j in B:
                total += D[i][j]
                count += 1
        return total / count if count else 0.0

    raise ValueError("linkage must be 'single', 'complete', or 'average'")
    # we chose not to use ward linkage here for simplicity



class AgglomerativeClustering(Unsupervised):
    """Agglomerative (hierarchical) clustering with single/complete/average linkage."""

    def __init__(self, n_clusters: int, linkage: str = "single", metric: str = "manhattan"):
        self.n_clusters = int(n_clusters)
        self.linkage = linkage
        self.metric = metric
        self.labels_ = None

    def fit(self, X, y=None):
        if X is None:
            raise ValueError("X cannot be None")
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("X cannot be empty")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than number of samples")

        linkage = self.linkage
        if linkage not in ("single", "complete", "average"):
            raise ValueError("linkage must be 'single', 'complete', or 'average'")

        dist_fn = _get_distance(self.metric)

        X_list = [_as_point(X[i]) for i in range(n_samples)]
        d = len(X_list[0])
        if d == 0:
            raise ValueError("Samples must have at least 1 feature")
        for i, p in enumerate(X_list):
            if len(p) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(p)}")

        # Precompute point-to-point distances once
        D = _pairwise_dist_matrix(X_list, dist_fn)

        # Start with singleton clusters (store indices)
        clusters: List[List[int]] = [[i] for i in range(n_samples)]

        # Merge until desired number of clusters
        while len(clusters) > self.n_clusters:
            best_i, best_j = 0, 1
            best_dist = _cluster_distance(clusters[0], clusters[1], D, linkage)

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d_ij = _cluster_distance(clusters[i], clusters[j], D, linkage)
                    if d_ij < best_dist:
                        best_dist = d_ij
                        best_i, best_j = i, j

            # Merge clusters best_i and best_j
            merged = clusters[best_i] + clusters[best_j]
            # Remove higher index first to avoid shifting
            if best_i < best_j:
                del clusters[best_j]
                del clusters[best_i]
            else:
                del clusters[best_i]
                del clusters[best_j]
            clusters.append(merged)

        # Build labels_
        labels = [-1] * n_samples
        for cid, idxs in enumerate(clusters):
            for idx in idxs:
                labels[idx] = cid

        self.labels_ = labels
        return self





# ---- Spectral Clustering ----
def _rbf_affinity(X_list: List[List[float]], dist_fn, gamma: float) -> List[List[float]]:
    """
    Fully-connected RBF affinity matrix:
    W_ij = exp(-gamma * dist(x_i, x_j))
    Note: if dist_fn is squared-euclidean, this is the common form.
    """

    n = len(X_list)
    W = [[0.0] * n for _ in range(n)]
    for i in range(n):
        W[i][i] = 0.0
        for j in range(i + 1, n):
            d = dist_fn(X_list[i], X_list[j])
            w = math.exp(-gamma * d)
            W[i][j] = w
            W[j][i] = w
    return W


def _row_normalize(M: List[List[float]]) -> List[List[float]]:
    """Normalize each row to unit length (Ng-Jordan-Weiss style)."""
    import math

    out = []
    for row in M:
        norm = math.sqrt(sum(v * v for v in row))
        if norm == 0.0:
            out.append([0.0 for _ in row])
        else:
            out.append([v / norm for v in row])
    return out



class SpectralClustering(Unsupervised):
    """
    Spectral clustering (simple version):
    1) Build affinity matrix W with RBF kernel
    2) Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    3) Take k smallest eigenvectors of L
    4) Row-normalize and run KMeans in that embedding space
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        gamma: float = 1.0,
        max_iter: int = 300,
    ):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.metric = "sqeuclidean"             # only euclidean for spectral clustering
        self.gamma = float(gamma)
        self.max_iter = int(max_iter)
        self.labels_ = None

    def fit(self, X, y=None):
        # ---- validation ----
        if X is None:
            raise ValueError("X cannot be None")
        n = len(X)
        if n == 0:
            raise ValueError("X cannot be empty")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.n_clusters > n:
            raise ValueError("n_clusters cannot be greater than number of samples")
        if self.gamma <= 0:
            raise ValueError("gamma must be > 0")

        # ---- convert X to list of points ----
        X_list = [_as_point(X[i]) for i in range(n)]
        d = len(X_list[0])
        if d == 0:
            raise ValueError("Samples must have at least 1 feature")
        for i, p in enumerate(X_list):
            if len(p) != d:
                raise ValueError(f"Inconsistent feature dimensions at index {i}: expected {d}, got {len(p)}")

        # ---- build affinity matrix W ----
        dist_fn = _get_distance(self.metric)
        W = _rbf_affinity(X_list, dist_fn, gamma=self.gamma)

        # ---- compute normalized Laplacian L = I - D^{-1/2} W D^{-1/2} ----
        try:
            import numpy as np      # for eigen decomposition
        except ImportError:
            raise ImportError("SpectralClustering requires numpy (for eigen decomposition).")

        Wn = np.array(W, dtype=float)
        degrees = Wn.sum(axis=1)
        # avoid division by zero
        degrees[degrees == 0.0] = 1.0
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

        I = np.eye(n)
        L = I - D_inv_sqrt @ Wn @ D_inv_sqrt  # symmetric

        # ---- eigenvectors for k smallest eigenvalues ----
        eigvals, eigvecs = np.linalg.eigh(L)
        U = eigvecs[:, : self.n_clusters]  # n x k

        # ---- row-normalize embedding ----
        U_list = U.tolist()
        U_norm = _row_normalize(U_list)

        # ---- cluster rows with KMeans ----
        km = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            metric="euclidean",  # kmeans in embedding space
        )
        km.fit(U_norm)

        self.labels_ = km.labels_
        return self
