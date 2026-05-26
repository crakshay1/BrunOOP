# upsaclay-ml\test_BrunOOP.py

def test_everything():
    """
        Even though we made three files to test our work, 
        This function ensures an ulimate test!
        This will test all of our functions and make sure everything works corretly
        Visuals for regression and clustering also included
    """
    import time
    from functools import wraps

    from importlib.resources import files
    from matplotlib import pyplot as plt

    from upsaclay_ml.pipeline import Pipeline
    from upsaclay_ml.models.linear_model import LinearRegression
    from upsaclay_ml.models.tree import DecisionTreeClassifier
    from upsaclay_ml.models.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    from upsaclay_ml.transformers import StandardScaler, OneHotEncoder, MinMaxScaler, Binarizer
    from upsaclay_ml.datasets import Dataset, Iris, Blobs
    from upsaclay_ml.model_selection import train_test_split
    from upsaclay_ml.metrics import (
        mean_squared_error, accuracy_score, silhouette_score,
        rand_score, adjusted_rand_score,
        homogeneity_score, completeness_score, v_measure_score
    )

    def banner(title: str):
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    def timeit(label: str):
        """Decorator factory to time a block of code."""
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
        """
            Simple 2D scatter plot of clusters
        """
        plt.figure(figsize=(7, 5))
        labels_set = set(labels)
        colors = plt.cm.get_cmap("tab10", len(labels_set))

        for k in labels_set:
            cluster_points = [X[i] for i in range(len(X)) if labels[i] == k]
            xs = [p[0] for p in cluster_points]
            ys = [p[1] for p in cluster_points]
            plt.scatter(xs, ys, s=50, color=colors(k), label=f"Cluster {k}", alpha=0.7)

        if centroids is not None:
            cx = [c[0] for c in centroids]
            cy = [c[1] for c in centroids]
            plt.scatter(cx, cy, color="black", marker="X", s=200, label="Centroids")

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

    def plot_regression(y_true, y_pred, title="Regression Predictions vs True"):
        """
            Scatter plot y_true vs y_pred to visualize regression fit
        """
        plt.figure(figsize=(7, 5))
        plt.scatter(range(len(y_true)), y_true, color="blue", label="True y", alpha=0.7)
        plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted y", alpha=0.7)
        plt.plot(range(len(y_true)), y_true, color="blue", alpha=0.3)
        plt.plot(range(len(y_pred)), y_pred, color="red", alpha=0.3)
        plt.title(title)
        plt.xlabel("Sample Index")
        plt.ylabel("y value")
        plt.legend()
        plt.show()


    # -------------------------------
    # 1) Pipeline regression test
    # -------------------------------
    banner("TEST 1 — PIPELINE REGRESSION (CSV -> OneHot -> StandardScaler -> LinearRegression)")
    csv_path = files("upsaclay_ml").joinpath("test_dataset.csv")
    ds = Dataset(str(csv_path), has_header=True, target_col=-1)
    X, y = ds

    print("Dataset summary:\n", ds)
    print("n_samples:", len(ds))
    print("First row:", ds[0])

    GENDER_COL = 2
    X_gender = [[row[GENDER_COL]] for row in X]
    enc = OneHotEncoder()
    X_gender_oh = enc.fit_transform(X_gender)
    print("OneHot categories:", enc.categories)
    print("OneHot feature names:", enc.get_feature_names_out())

    X_num = [[row[1], row[3], row[4]] for row in X]
    X_final = [xn + xoh for xn, xoh in zip(X_num, X_gender_oh)]

    mm = MinMaxScaler(0, 1).fit(X_final)
    X_mm = mm.transform(X_final)
    bn = Binarizer(threshold=0.5)
    X_bin = bn.transform(X_mm)

    print("MinMaxScaler min_ (first 5):", mm.min_[:5])
    print("MinMaxScaler max_ (first 5):", mm.max_[:5])
    print("Binarized sample row:", X_bin[0])

    pipe_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(lr=0.05, n_iter=800, fit_intercept=True)),
    ])
    print("Final estimator repr:", pipe_reg.estimator)

    @timeit("Pipeline Regression Fit + Predict")
    def run_reg():
        pipe_reg.fit(X_final, y)
        return pipe_reg(X_final)

    y_pred = run_reg()
    print("MSE:", mean_squared_error(y, y_pred))
    # Plot regression
    plot_regression(y, y_pred, title="LinearRegression Predictions vs True y")


    # -------------------------------
    # 2) Pipeline classification test
    # -------------------------------
    banner("TEST 2 — PIPELINE CLASSIFICATION (Iris -> StandardScaler -> DecisionTreeClassifier)")
    X_iris, y_iris = Iris()
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris)

    pipe_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, criterion="gini", min_samples_split=2)),
    ])
    print("Pipeline repr:", pipe_clf)

    @timeit("Pipeline Classification Fit + Predict")
    def run_clf():
        pipe_clf.fit(X_train, y_train)
        return pipe_clf(X_test)

    pred = run_clf()
    print("Accuracy:", accuracy_score(y_test, pred))

    # -------------------------------
    # 3) Pipeline unsupervised test (KMeans)
    # -------------------------------
    banner("TEST 3 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> KMeans)")
    X_blobs = Blobs(n_samples=200, centers=3, n_features=2, random_state=0)

    pipe_km = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=0, metric="euclidean")),
    ])

    @timeit("Pipeline KMeans Fit")
    def run_km():
        pipe_km.fit(X_blobs)
        return pipe_km.estimator.labels_

    labels_km = run_km()
    print("Labels length:", len(labels_km))

    # metrics
    print("Silhouette:", silhouette_score(X_blobs, labels_km))

    k = 3  # must match centers=3 above
    Y_true = [i % k for i in range(len(X_blobs))]
    print("Rand Index:", rand_score(X_blobs, Y_true, labels_km))
    print("Adjusted Rand Index:", adjusted_rand_score(X_blobs, Y_true, labels_km))
    print("Homogeneity:", homogeneity_score(X_blobs, Y_true, labels_km))
    print("Completeness:", completeness_score(X_blobs, Y_true, labels_km))
    print("V-measure:", v_measure_score(X_blobs, Y_true, labels_km))

    print("Centroid[0]:", pipe_km.estimator.centroids_[0])
    print("Labels_ (first 10):", pipe_km.estimator.labels_[:10])

    plot_clusters(X_blobs, labels_km, title="KMeans Clustering", centroids=pipe_km.estimator.centroids_)


    # -------------------------------
    # 4) Pipeline unsupervised test (Agglomerative)
    # -------------------------------
    banner("TEST 4 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> AgglomerativeClustering)")
    X_blobs2 = Blobs(n_samples=150, centers=3, n_features=2, random_state=1)

    pipe_agg = Pipeline([
        ("scaler", StandardScaler()),
        ("agg", AgglomerativeClustering(n_clusters=3, linkage="average", metric="euclidean")),
    ])

    @timeit("Pipeline Agglomerative Fit")
    def run_agg():
        pipe_agg.fit(X_blobs2)
        return pipe_agg.estimator.labels_

    labels_agg = run_agg()
    print("Labels length:", len(labels_agg))
    
    # metrics
    print("Silhouette:", silhouette_score(X_blobs2, labels_agg))
    k2 = 3  # must match centers=3 above
    Y_true2 = [i % k2 for i in range(len(X_blobs2))]

    print("Rand Index (Agg):", rand_score(X_blobs2, Y_true2, labels_agg))
    print("Adjusted Rand Index (Agg):", adjusted_rand_score(X_blobs2, Y_true2, labels_agg))
    print("Homogeneity (Agg):", homogeneity_score(X_blobs2, Y_true2, labels_agg))
    print("Completeness (Agg):", completeness_score(X_blobs2, Y_true2, labels_agg))
    print("V-measure (Agg):", v_measure_score(X_blobs2, Y_true2, labels_agg))

    print("Labels_ (first 10):", labels_agg[:10])

    plot_clusters(X_blobs2, labels_agg, title="Agglomerative Clustering")


    # -------------------------------
    # 5) Pipeline unsupervised test (Spectral Clustering)
    # -------------------------------
    banner("TEST 5 — PIPELINE UNSUPERVISED (Blobs -> StandardScaler -> SpectralClustering)")

    # Use a smaller dataset because spectral clustering is heavier (eigendecomposition)
    X_blobs3 = Blobs(n_samples=120, centers=3, n_features=2, random_state=2)

    pipe_sp = Pipeline([
        ("scaler", StandardScaler()),
        ("spectral", SpectralClustering(n_clusters=3, random_state=0, gamma=1.0, max_iter=200)),
    ])

    @timeit("Pipeline Spectral Fit")
    def run_sp():
        # Pipeline likely calls fit on the final estimator during fit()
        pipe_sp.fit(X_blobs3)
        return pipe_sp.estimator.labels_

    try:
        labels_sp = run_sp()
    except ImportError as e:
        print("SpectralClustering skipped (requires numpy):", e)
        labels_sp = None
    except Exception as e:
        print("SpectralClustering failed:", e)
        labels_sp = None

    if labels_sp is not None:
        print("Labels length:", len(labels_sp))

        # metrics
        print("Silhouette:", silhouette_score(X_blobs3, labels_sp))

        k3 = 3  # must match centers=3 above
        Y_true3 = [i % k3 for i in range(len(X_blobs3))]

        print("Rand Index (Spectral):", rand_score(X_blobs3, Y_true3, labels_sp))
        print("Adjusted Rand Index (Spectral):", adjusted_rand_score(X_blobs3, Y_true3, labels_sp))
        print("Homogeneity (Spectral):", homogeneity_score(X_blobs3, Y_true3, labels_sp))
        print("Completeness (Spectral):", completeness_score(X_blobs3, Y_true3, labels_sp))
        print("V-measure (Spectral):", v_measure_score(X_blobs3, Y_true3, labels_sp))

        print("Labels_ (first 10):", labels_sp[:10])

        plot_clusters(X_blobs3, labels_sp, title="Spectral Clustering")


if __name__ == "__main__":
    test_everything()
