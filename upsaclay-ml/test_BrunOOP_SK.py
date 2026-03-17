# upsaclay-ml/test_BrunOOP_SK.py

def test_everything_sklearn():
    """
    Sklearn version of the ultimate pipeline test
    (same tests, same metrics, same timing, same style as your custom version)
    """
    import time
    from functools import wraps
    from importlib.resources import files

    import numpy as np
    import pandas as pd

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, Binarizer
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

    from sklearn.metrics import (
        mean_squared_error,
        accuracy_score,
        silhouette_score,
        rand_score,
        adjusted_rand_score,
        homogeneity_score,
        completeness_score,
        v_measure_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris, make_blobs

    from matplotlib import pyplot as plt

    def banner(title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def timeit(label: str):
        def deco(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                dt = time.perf_counter() - t0
                print(f"[TIME] {label}: {dt:.6f} s")
                return out
            return wrapper
        return deco

    def plot_clusters(X, labels, title="Cluster Plot", centroids=None):
        plt.figure(figsize=(7, 5))
        labels_set = set(labels)
        colors = plt.get_cmap("tab10", len(labels_set))

        for k in labels_set:
            cluster_points = X[labels == k]
            xs = cluster_points[:, 0]
            ys = cluster_points[:, 1]
            plt.scatter(xs, ys, s=50, color=colors(k), label=f"Cluster {k}", alpha=0.7)

        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="X", s=200, label="Centroids")

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def plot_regression(y_true, y_pred, title="Regression Predictions vs True"):
        plt.figure(figsize=(7, 5))
        plt.scatter(range(len(y_true)), y_true, label="True y", alpha=0.7)
        plt.scatter(range(len(y_pred)), y_pred, label="Predicted y", alpha=0.7)
        plt.plot(range(len(y_true)), y_true, alpha=0.3)
        plt.plot(range(len(y_pred)), y_pred, alpha=0.3)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("y value")
        plt.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # 1) Pipeline regression test (CSV -> OneHot -> StandardScaler -> LinearRegression)
    # -------------------------------------------------------------------------
    banner("TEST 1 — PIPELINE REGRESSION (CSV -> OneHot -> StandardScaler -> LinearRegression)")

    csv_path = files("upsaclay_ml").joinpath("test_dataset.csv")
    df = pd.read_csv(csv_path)

    # mimic your custom dataset: drop id, onehot gender, keep numeric
    # Your custom: X: [id, age, gender, height_cm, weight_kg], y=label
    X_df = df[["age", "gender", "height_cm", "weight_kg"]].copy()
    y = df["label"].to_numpy()

    # One-hot info (to print like your script)
    # compatibility: sparse vs sparse_output depending on sklearn version
    try:
        enc_for_print = OneHotEncoder(sparse=False)
    except TypeError:
        enc_for_print = OneHotEncoder(sparse_output=False)

    X_gender = X_df[["gender"]].to_numpy()
    X_gender_oh = enc_for_print.fit_transform(X_gender)
    print("OneHot categories:", enc_for_print.categories_)
    print("OneHot feature names:", enc_for_print.get_feature_names_out(["gender"]))

    # numeric + onehot (to print MinMax/Binarizer results like yours)
    X_num = X_df[["age", "height_cm", "weight_kg"]].to_numpy()
    X_final = np.hstack([X_num, X_gender_oh])

    mm = MinMaxScaler(feature_range=(0, 1))
    X_mm = mm.fit_transform(X_final)
    bn = Binarizer(threshold=0.5)
    X_bin = bn.fit_transform(X_mm)

    # Print like your custom version
    print("MinMaxScaler min_ (first 5):", X_final.min(axis=0)[:5])
    print("MinMaxScaler max_ (first 5):", X_final.max(axis=0)[:5])
    print("Binarized sample row:", X_bin[0].astype(int).tolist())

    # Build a real sklearn pipeline: preprocess -> model
    # ColumnTransformer: scale numeric + onehot gender then linear regression
    numeric_cols = ["age", "height_cm", "weight_kg"]
    cat_cols = ["gender"]

    try:
        onehot = OneHotEncoder(sparse=False)
    except TypeError:
        onehot = OneHotEncoder(sparse_output=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", onehot, cat_cols),
        ],
        remainder="drop",
    )

    pipe_reg = Pipeline([
        ("preprocess", pre),
        ("model", LinearRegression()),
    ])

    print("Final estimator repr:", pipe_reg.named_steps["model"])

    @timeit("Pipeline Regression Fit + Predict")
    def run_reg():
        pipe_reg.fit(X_df, y)
        return pipe_reg.predict(X_df)

    y_pred = run_reg()
    print("MSE:", mean_squared_error(y, y_pred))

    plot_regression(y, y_pred, title="Sklearn LinearRegression Predictions vs True y")

    # -------------------------------------------------------------------------
    # 2) Pipeline classification test (Iris -> StandardScaler -> DecisionTreeClassifier)
    # -------------------------------------------------------------------------
    banner("TEST 2 — PIPELINE CLASSIFICATION (Iris -> StandardScaler -> DecisionTreeClassifier)")

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

    pipe_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, criterion="gini", min_samples_split=2, random_state=42)),
    ])

    print("Pipeline repr:", pipe_clf)

    @timeit("Pipeline Classification Fit + Predict")
    def run_clf():
        pipe_clf.fit(X_train, y_train)
        return pipe_clf.predict(X_test)

    pred = run_clf()
    print("Accuracy:", accuracy_score(y_test, pred))

    # -------------------------------------------------------------------------
    # 3) Pipeline unsupervised test (Blobs -> StandardScaler -> KMeans)
    #    Here we use make_blobs so we get TRUE labels for rand/ARI/homogeneity...
    # -------------------------------------------------------------------------
    banner("TEST 3 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> KMeans)")

    X_blobs, Y_true = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)

    pipe_km = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=0, n_init=10)),
    ])

    @timeit("Pipeline KMeans Fit")
    def run_km():
        pipe_km.fit(X_blobs)
        return pipe_km.named_steps["kmeans"].labels_

    labels_km = run_km()
    print("Labels length:", len(labels_km))

    # NOTE: sklearn silhouette should be computed on the SAME space you clustered in.
    X_blobs_scaled = pipe_km.named_steps["scaler"].transform(X_blobs)

    print("Silhouette:", silhouette_score(X_blobs_scaled, labels_km))
    print("Rand Index:", rand_score(Y_true, labels_km))
    print("Adjusted Rand Index:", adjusted_rand_score(Y_true, labels_km))
    print("Homogeneity:", homogeneity_score(Y_true, labels_km))
    print("Completeness:", completeness_score(Y_true, labels_km))
    print("V-measure:", v_measure_score(Y_true, labels_km))

    centroids = pipe_km.named_steps["kmeans"].cluster_centers_
    print("Centroid[0]:", centroids[0].tolist())
    print("Labels_ (first 10):", labels_km[:10].tolist())

    plot_clusters(X_blobs, labels_km, title="Sklearn KMeans Clustering", centroids=centroids)

    # -------------------------------------------------------------------------
    # 4) Pipeline unsupervised test (Blobs -> StandardScaler -> AgglomerativeClustering)
    # -------------------------------------------------------------------------
    banner("TEST 4 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> AgglomerativeClustering)")

    X_blobs2, Y_true2 = make_blobs(n_samples=150, centers=3, n_features=2, random_state=1)

    # For Agglomerative, no .labels_ until after fit, and it's not always pipeline-friendly for labels,
    # but we can still use sklearn Pipeline for scaling and then fit clustering on transformed data.
    scaler2 = StandardScaler()

    @timeit("Pipeline Agglomerative Fit")
    def run_agg():
        X_scaled = scaler2.fit_transform(X_blobs2)
        agg = AgglomerativeClustering(n_clusters=3, linkage="average")
        labels = agg.fit_predict(X_scaled)
        return X_scaled, labels

    X_blobs2_scaled, labels_agg = run_agg()
    print("Labels length:", len(labels_agg))

    print("Silhouette:", silhouette_score(X_blobs2_scaled, labels_agg))
    print("Rand Index (Agg):", rand_score(Y_true2, labels_agg))
    print("Adjusted Rand Index (Agg):", adjusted_rand_score(Y_true2, labels_agg))
    print("Homogeneity (Agg):", homogeneity_score(Y_true2, labels_agg))
    print("Completeness (Agg):", completeness_score(Y_true2, labels_agg))
    print("V-measure (Agg):", v_measure_score(Y_true2, labels_agg))

    print("Labels_ (first 10):", labels_agg[:10].tolist())
    plot_clusters(X_blobs2, labels_agg, title="Sklearn Agglomerative Clustering")


    # -------------------------------------------------------------------------
    # 5) Pipeline unsupervised test (Blobs -> StandardScaler -> SpectralClustering)
    # -------------------------------------------------------------------------
    banner("TEST 5 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> SpectralClustering)")

    X_blobs3, Y_true3 = make_blobs(n_samples=120, centers=3, n_features=2, random_state=2)

    pipe_sp = Pipeline([
        ("scaler", StandardScaler()),
        ("spectral", SpectralClustering(
            n_clusters=3,
            random_state=0,
            affinity="rbf",
            gamma=1.0,
            assign_labels="kmeans"
        )),
    ])

    @timeit("Pipeline Spectral Fit")
    def run_sp():
        pipe_sp.fit(X_blobs3)
        # SpectralClustering stores labels_ after fit
        return pipe_sp.named_steps["spectral"].labels_

    labels_sp = run_sp()
    print("Labels length:", len(labels_sp))

    # Compute metrics in the SAME space used for clustering (scaled)
    X_blobs3_scaled = pipe_sp.named_steps["scaler"].transform(X_blobs3)

    print("Silhouette:", silhouette_score(X_blobs3_scaled, labels_sp))
    print("Rand Index (Spectral):", rand_score(Y_true3, labels_sp))
    print("Adjusted Rand Index (Spectral):", adjusted_rand_score(Y_true3, labels_sp))
    print("Homogeneity (Spectral):", homogeneity_score(Y_true3, labels_sp))
    print("Completeness (Spectral):", completeness_score(Y_true3, labels_sp))
    print("V-measure (Spectral):", v_measure_score(Y_true3, labels_sp))

    print("Labels_ (first 10):", labels_sp[:10].tolist())

    plot_clusters(X_blobs3, labels_sp, title="Sklearn Spectral Clustering")


if __name__ == "__main__":
    test_everything_sklearn()
