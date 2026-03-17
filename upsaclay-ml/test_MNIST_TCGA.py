# test_MNIST_TCGA.py

# test_mnist_small
def test_mnist_small():
    import time
    from upsaclay_ml.datasets import MNIST
    from upsaclay_ml.pipeline import Pipeline
    from upsaclay_ml.transformers import StandardScaler
    from upsaclay_ml.models.tree import DecisionTreeClassifier
    from upsaclay_ml.metrics import accuracy_score

    def take(ds, n):
        X, y = ds  # Dataset yields (data, target)
        return X[:n], y[:n]

    print("\n" + "=" * 80)
    print("MNIST SMALL TEST (StandardScaler -> DecisionTreeClassifier)")
    print("=" * 80)

    mnist_train = MNIST(split="train", normalize=True)
    mnist_test = MNIST(split="test", normalize=True)

    X_train, y_train = take(mnist_train, 2000)
    X_test, y_test = take(mnist_test, 500)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=10)),
    ])

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    pred = pipe(X_test)
    dt = time.perf_counter() - t0

    print("Train size:", len(X_train), " Test size:", len(X_test))
    print("Accuracy:", accuracy_score(y_test, pred))
    print(f"Time (fit+predict): {dt:.3f} s")



# test_tcga_small
def test_tcga_small():
    import time
    from upsaclay_ml.datasets import TCGA
    from upsaclay_ml.transformers import OneHotEncoder, StandardScaler
    from upsaclay_ml.pipeline import Pipeline
    from upsaclay_ml.models.cluster import KMeans
    from upsaclay_ml.metrics import silhouette_score

    print("\n" + "=" * 80)
    print("TCGA SMALL TEST (OneHot -> StandardScaler -> KMeans)")
    print("=" * 80)

    ds = TCGA(project="TCGA-BRCA", size=120)
    X, _ = ds  # target is None

    print("Dataset summary:\n", ds)
    print("First row:", X[0])

    # We'll cluster on categorical fields only (skip ids):
    # columns: [case_id, submitter_id, project_id, primary_site, disease_type]
    # take columns 2..4
    X_cat = [[row[2], row[3], row[4]] for row in X]

    # OneHotEncoder expects 2D; yours works with list rows
    enc = OneHotEncoder()
    X_oh = enc.fit_transform(X_cat)

    print("OneHot feature names (first 15):", enc.get_feature_names_out()[:15])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=3, random_state=0, metric="euclidean")),
    ])

    t0 = time.perf_counter()
    pipe.fit(X_oh)
    labels = pipe.estimator.labels_
    dt = time.perf_counter() - t0

    print("Labels length:", len(labels))
    print("Silhouette:", silhouette_score(X_oh, labels))
    print(f"Time (fit): {dt:.3f} s")


if __name__ == "__main__":
    test_mnist_small()
    test_tcga_small()
    
