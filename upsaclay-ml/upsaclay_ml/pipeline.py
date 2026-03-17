# upsaclay-ml/upsaclay_ml/pipeline.py

class Pipeline:
    """
        Simple pipeline for chaining transformers and a final estimator
        Each step is a tuple: (name, transformer/estimator)
    """

    def __init__(self, steps):
        self.steps = steps
        self.transformers = steps[:-1]  # all but last
        self.estimator = steps[-1][1]  # the last step object

    def fit(self, X, y=None):
        """
            Fit all transformers and then the final estimator
        """
        Xt = X
        for name, step in self.transformers:
            step.fit(Xt)
            Xt = step.transform(Xt)

        # Fit the final estimator
        if y is not None:
            self.estimator.fit(Xt, y)
        else:
            self.estimator.fit(Xt)
        return self

    def predict(self, X):
        """
            Transform X through all transformers, then call estimator.predict
        """
        Xt = X
        for name, step in self.transformers:
            Xt = step.transform(Xt)
        return self.estimator.predict(Xt)

    # Optional: allow calling pipeline like a function
    def __call__(self, X):
        return self.predict(X)

