# upsaclay-ml/upsaclay_ml/models/estimator.py


class Estimator():

    # Abstract fit method    
    def fit(self, X, y=None):
        raise NotImplementedError("fit method must be implemented by subclasses")
    
    # Compare estimators by class + hyperparameters. Ignore attributes ending with '_'.
    def __eq__(self, other):
        if type(self) is type(other):
            if {k_self: v_self for k_self, v_self in self.__dict__.items() if not k_self.endswith('_')} == \
            {k_other: v_other for k_other, v_other in other.__dict__.items() if not k_other.endswith('_')}:
                return True
        return False
        # or raise NotImplementedError("")

    # Call method to make the estimator callable
    def __call__(self, X, y=None):
        """
        Supervised: calls predict(X)
        Unsupervised: calls fit_predict(X)
        Transformer: calls transform(X)
        """
        if hasattr(self, "fit_predict"):
            return self.fit_predict(X, y)       # unsupervised
        if hasattr(self, "predict"):
            return self.predict(X)              # supervised
        if hasattr(self, "transform"):
            return self.transform(X)            # transformers (optional)
        raise TypeError(f"{self.__class__.__name__} is not callable")
    
    # String representation (class + hyperparameters).
    def __repr__(self):
        return f"{self.__class__.__name__}"\
        f"({', '.join(f'{k}={v}' for k, v in self.__dict__.items() if not k.endswith('_'))})"
    
    def __str__(self):
        return self.__repr__()
    



class Unsupervised(Estimator):

    # Fit the model and return cluster labels
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        if not hasattr(self, 'labels_'):
            raise AttributeError("Subclasses must set 'labels_' attribute during fit")
        return self.labels_         # Assumes subclasses set labels_ during fit

    def __call__(self, X, y=None):
        return self.fit_predict(X, y)
    



class Supervised(Estimator):

    # Abstract predict method
    def predict(self, X, y=None):
        raise NotImplementedError("predict method must be implemented by subclasses")

    def __call__(self, X, y=None):
        return self.predict(X)
    

