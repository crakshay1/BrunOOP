# upsaclay_ml/datasets.py

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import random
import struct
import urllib.request
from typing import Any, Iterator


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _download_if_missing(url: str, dst_path: str) -> None:
    if os.path.exists(dst_path):
        return
    _ensure_dir(os.path.dirname(dst_path))
    urllib.request.urlretrieve(url, dst_path)


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _auto_cast(v: str) -> Any:
    """Try int -> float -> else keep string."""
    s = v.strip()
    if s == "":
        return ""  # or None if you prefer
    try:
        iv = int(s)
        return iv
    except Exception:
        pass
    try:
        fv = float(s)
        return fv
    except Exception:
        return s


class Dataset:
    """
    One unified dataset class.

    Can be constructed as:
      - Dataset(path_or_url: str, target_col=..., has_header=...)
      - Dataset(X: list[list[Any]], y: list | None)
    """

    def __init__(
        self,
        X_or_path: Any,
        y: Any = None,
        *,
        target_col: int | None = -1,
        has_header: bool = True,
        delimiter: str = ",",
        encoding: str = "utf-8",
        name: str = "Dataset",
    ):
        self.name = name
        self.path: str | None = None
        self.url: str | None = None

        # IMPORTANT: do NOT build self.data from X here (X may not exist yet)
        self.data: list[list[Any]] | None = None
        self.target: list[Any] | None = None
        self.header: list[str] | None = None

        self._has_header = bool(has_header)
        self._delimiter = delimiter
        self._encoding = encoding
        self._target_col = target_col

        # Case A: CSV path/URL
        if isinstance(X_or_path, str):
            if _is_url(X_or_path):
                self.url = X_or_path
            else:
                self.path = X_or_path
            self.load()
            return

        # Case B: in-memory X/y
        X = X_or_path
        self.data = [list(row) for row in X]  # keep raw values
        self.target = None if y is None else list(y)

        if self.target is not None and len(self.data) != len(self.target):
            raise ValueError("X and y must have the same length")


    def load(self) -> "Dataset":
        """Load CSV from self.path or self.url into (data, target)."""
        if self.path is None and self.url is None:
            raise ValueError("No path or url set for this Dataset")

        # read text
        if self.url is not None:
            with urllib.request.urlopen(self.url) as resp:
                text = resp.read().decode(self._encoding)
            f = io.StringIO(text)
        else:
            f = open(self.path, "r", encoding=self._encoding, newline="")

        try:
            reader = csv.reader(f, delimiter=self._delimiter)
            rows = [r for r in reader if r]
        finally:
            f.close()

        if not rows:
            raise ValueError("CSV has no rows")

        # header
        start = 0
        if self._has_header:
            self.header = rows[0]
            start = 1

        # split X/y using target_col
        if self._target_col is None:
            X_rows = rows[start:]
            self.data = [[_auto_cast(v) for v in r] for r in X_rows]
            self.target = None
            return self

        X_out: list[list[Any]] = []
        y_out: list[Any] = []

        for r in rows[start:]:
            if len(r) == 0:
                continue
            tc = self._target_col if self._target_col >= 0 else (len(r) + self._target_col)
            if tc < 0 or tc >= len(r):
                raise ValueError(f"target_col {self._target_col} is out of range for row with {len(r)} cols")

            y_out.append(_auto_cast(r[tc]))
            x_row = [_auto_cast(v) for j, v in enumerate(r) if j != tc]
            X_out.append(x_row)

        self.data = X_out
        self.target = y_out
        return self

    def __len__(self) -> int:
        return 0 if self.data is None else len(self.data)

    def __getitem__(self, idx):
        if self.data is None:
            raise ValueError("Dataset is not loaded.")
        if isinstance(idx, slice):
            Xs = self.data[idx]
            ys = None if self.target is None else self.target[idx]
            return Dataset(Xs, ys, name=f"{self.name}[{idx}]")
        if self.target is None:
            return self.data[idx]
        return self.data[idx], self.target[idx]

    def __iter__(self) -> Iterator:
        """
        Enables: X, y = Dataset("file.csv")
        by yielding exactly (data, target).
        """
        if self.data is None:
            raise ValueError("Dataset is not loaded.")
        yield self.data
        yield self.target

    def __str__(self) -> str:
        n_rows = len(self)
        n_cols = len(self.data[0]) if self.data else 0
        src = self.path or self.url or "in-memory"
        tgt = "yes" if self.target is not None else "no"
        hdr = "yes" if self.header is not None else "no"
        return (
            f"{self.name}\n"
            f"source: {src}\n"
            f"rows: {n_rows}\n"
            f"cols: {n_cols}\n"
            f"target: {tgt}\n"
            f"header: {hdr}"
        )

    def __repr__(self) -> str:
        return f"Dataset(name={self.name!r}, n={len(self)})"


# ---------------------------
# Blobs dataset (synthetic)
# ---------------------------

class BlobsDataset(Dataset):
    def __init__(
        self,
        n_samples: int = 100,
        centers: int = 3,
        n_features: int = 2,
        center_box: tuple[float, float] = (-10.0, 10.0),
        cluster_std: float = 1.0,
        random_state: int | None = None,
        name: str = "Blobs",
    ):
        self.n_samples = int(n_samples)
        self.centers = int(centers)
        self.n_features = int(n_features)
        self.center_box = center_box
        self.cluster_std = float(cluster_std)
        self.random_state = random_state

        # initialize empty base dataset
        super().__init__([], None, name=name)

        self.load()

    def load(self) -> "BlobsDataset":
        if self.n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if self.centers <= 0:
            raise ValueError("centers must be > 0")
        if self.n_features <= 0:
            raise ValueError("n_features must be > 0")
        if self.cluster_std <= 0:
            raise ValueError("cluster_std must be > 0")

        rng = random.Random(self.random_state)
        low, high = self.center_box

        C = [
            [rng.uniform(low, high) for _ in range(self.n_features)]
            for _ in range(self.centers)
        ]

        X = []
        for i in range(self.n_samples):
            c = C[i % self.centers]
            point = [rng.gauss(mu, self.cluster_std) for mu in c]
            X.append(point)

        self.data = X
        self.target = None
        return self


def Blobs(
    n_samples: int = 100,
    centers: int = 3,
    n_features: int = 2,
    center_box: tuple[float, float] = (-10.0, 10.0),
    cluster_std: float = 1.0,
    random_state: int | None = None,
):
    """Convenience function used in your unsupervised_learning.py. Returns X."""
    ds = BlobsDataset(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        center_box=center_box,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return ds.data


# ---------------------------
# Iris dataset (download + cache)
# ---------------------------

class IrisDataset(Dataset):
    def __init__(
        self,
        root: str = ".cache/upsaclay_ml",
        download: bool = True,
        name: str = "Iris",
    ):
        self.root = root
        self.download = bool(download)
        super().__init__([], [], name=name)
        self.load()

    def load(self) -> "IrisDataset":
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        cache_dir = os.path.join(self.root, "iris")
        _ensure_dir(cache_dir)
        path = os.path.join(cache_dir, "iris.csv")

        if self.download:
            _download_if_missing(url, path)

        X: list[list[float]] = []
        y: list[int] = []

        label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                X.append([
                    float(row["sepal_length"]),
                    float(row["sepal_width"]),
                    float(row["petal_length"]),
                    float(row["petal_width"]),
                ])
                y.append(label_map[row["species"]])

        self.data = X
        self.target = y
        return self


def Iris(root: str = ".cache/upsaclay_ml", download: bool = True):
    """Convenience function used in your supervised_learning.py. Returns (X, y)."""
    ds = IrisDataset(root=root, download=download)
    return ds.data, ds.target


# ---------------------------
# MNIST dataset (download + cache)
# ---------------------------

def _mnist_read_images_gz(path: str) -> list[list[int]]:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad MNIST image magic {magic} in {path}")
        size = rows * cols
        imgs = []
        for _ in range(n):
            imgs.append(list(f.read(size)))
        return imgs


def _mnist_read_labels_gz(path: str) -> list[int]:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad MNIST label magic {magic} in {path}")
        return list(f.read(n))


class MNISTDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        root: str = ".cache/upsaclay_ml",
        download: bool = True,
        normalize: bool = False,
        name: str | None = None,
    ):
        self.split = split.lower()
        self.root = root
        self.download = bool(download)
        self.normalize = bool(normalize)

        ds_name = name if name is not None else f"MNIST[{self.split}]"
        super().__init__([], [], name=ds_name)

        self.load()

    def load(self) -> "MNISTDataset":
        if self.split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        base = "https://raw.githubusercontent.com/fgnt/mnist/master/"
        urls = {
            "train_images": base + "train-images-idx3-ubyte.gz",
            "train_labels": base + "train-labels-idx1-ubyte.gz",
            "test_images":  base + "t10k-images-idx3-ubyte.gz",
            "test_labels":  base + "t10k-labels-idx1-ubyte.gz",
        }

        cache_dir = os.path.join(self.root, "mnist")
        _ensure_dir(cache_dir)

        if self.split == "train":
            img_url, lbl_url = urls["train_images"], urls["train_labels"]
            img_fn,  lbl_fn  = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
        else:
            img_url, lbl_url = urls["test_images"], urls["test_labels"]
            img_fn,  lbl_fn  = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"

        img_path = os.path.join(cache_dir, img_fn)
        lbl_path = os.path.join(cache_dir, lbl_fn)

        if self.download:
            _download_if_missing(img_url, img_path)
            _download_if_missing(lbl_url, lbl_path)

        X = _mnist_read_images_gz(img_path)
        y = _mnist_read_labels_gz(lbl_path)

        if self.normalize:
            X = [[px / 255.0 for px in row] for row in X]
        else:
            # still cast to float for consistency with models
            X = [[float(px) for px in row] for row in X]

        self.data = X
        self.target = y
        return self


def MNIST(
    split: str = "train",
    root: str = ".cache/upsaclay_ml",
    download: bool = True,
    normalize: bool = False,
    name: str | None = None,
):
    """Convenience function: returns a Dataset-like object."""
    return MNISTDataset(split=split, root=root, download=download, normalize=normalize, name=name)


# ---------------------------
# TCGA dataset (API -> CSV-like table, but returned as Dataset)
# ---------------------------

class TGCADataset(Dataset):
    def __init__(
        self,
        project: str = "TCGA-BRCA",
        size: int = 200,
        name: str | None = None,
    ):
        self.project = project
        self.size = int(size)
        ds_name = name if name is not None else f"TCGA[{project}]-cases"
        super().__init__([], None, name=ds_name)
        self.load()

    def load(self) -> "TGCADataset":
        endpoint = "https://api.gdc.cancer.gov/cases"
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "project.project_id", "value": [self.project]},
            },
            "fields": ",".join([
                "case_id",
                "submitter_id",
                "project.project_id",
                "primary_site",
                "disease_type",
            ]),
            "format": "JSON",
            "size": int(self.size),
        }

        resp = _post_json(endpoint, payload)
        hits = resp.get("data", {}).get("hits", [])

        # TCGA is categorical strings, so store as raw strings in X (not float)
        X: list[list[Any]] = []
        y = None

        for h in hits:
            proj = (h.get("project") or {}).get("project_id", "")
            X.append([
                h.get("case_id", ""),
                h.get("submitter_id", ""),
                proj,
                h.get("primary_site", ""),
                h.get("disease_type", ""),
            ])

        # We keep target None because this endpoint doesn't give labels for supervised learning.
        self.data = X  # type: ignore
        self.target = y
        self.header = ["case_id", "submitter_id", "project_id", "primary_site", "disease_type"]
        return self


def TCGA(project: str = "TCGA-BRCA", size: int = 200, name: str | None = None):
    """Convenience function: returns a Dataset-like object (categorical table)."""
    return TGCADataset(project=project, size=size, name=name)
