# upsaclay-ml/upsaclay_ml/transformers.py

from .models.estimator import Estimator

class Transformer(Estimator):
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        """
            Fits then transforms X data
        """
        self.fit(X)
        return self.transform(X)
    
    def fit(self, X):
        """
            Learns how to transform
        """
        return self



class Binarizer(Transformer):
    """
        Converts numerical values to 0 or 1 based on a threshold
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def transform(self, X):
        return [[1 if x > self.threshold else 0 for x in row] for row in X]
        

class StandardScaler(Transformer):
    """
        Standardizes features to mean 0 and standard deviation 1
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
            Computes mean and standard deviation for each feature
        """
        n_features = len(X[0])
        self.mean_ = []
        self.std_ = []

        for j in range(n_features):
            column = [row[j] for row in X]
            mean = sum(column) / len(column)
            variance = sum((x - mean) ** 2 for x in column) / len(column)
            # Basically here we are computing the mean and variance of a column

            self.mean_.append(mean)
            self.std_.append(variance ** 0.5)

        return self

    def transform(self, X):
        """
            Apply standardization using learned mean and std
        """
        transformed = []
        for row in X:
            scaled_row = []
            for j in range(len(row)):
                x = row[j] # x value in a given row
                mean = self.mean_[j] # Mean of the column
                std = self.std_[j] # std of the column

                scaled_value = (x - mean) / (std + 1e-8) 
                # As we want to avoid division by 0, we added 1e-8
                scaled_row.append(scaled_value)

            transformed.append(scaled_row) 
            # New X man (You guys got it? like X-men haha... (subscribe to my Youtube channel))

        return transformed


class MinMaxScaler(Transformer):
    """
        Scales features to a fixed range [minimum, maximum]
    """
    def __init__(self, minimum=0, maximum=1):
        self.minimum = minimum
        self.maximum = maximum
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        """
            Computes min and max values for each feature
        """
        n_features = len(X[0])
        self.min_ = []
        self.max_ = []

        for j in range(n_features):
            column = [row[j] for row in X]
            self.min_.append(min(column))
            self.max_.append(max(column))

        return self

    def transform(self, X):
        """
            Apply Min-Max scaling using learned min and max
        """
        transformed = []
        for row in X:
            min_maxed_row = []
            for j in range(len(row)):
                x = row[j]
                min_maxed = (self.maximum - self.minimum) * (x - self.min_[j]) / (self.max_[j] - self.min_[j] + 1e-8) + self.minimum
                min_maxed_row.append(min_maxed) # It's kinda the same logic as StandardScaler
            transformed.append(min_maxed_row)
        
        return transformed


class OneHotEncoder(Transformer):
    """
        Encodes categorical values as one-hot vectors
    """

    def __init__(self, categories=None):
        self.categories = categories if categories else []

    def fit(self, X):
        """
            Learns unique categories from X if not provided
        """
        if not self.categories:
            unique = set()
            for row in X:
                for val in row:
                    unique.add(val)
            self.categories = sorted(unique)

        return self

    def transform(self, X):
        """
            Converts categorical values to one-hot encoded vectors
        """
        transformed = []
        # I'm the one who is hot (One-hot guys)
        for row in X:
            encoded_row = []
            for val in row:
                encoded_row.extend(
                    [1 if val == cat else 0 for cat in self.categories]
                ) # Extend is adding elements at the end of the list
            transformed.append(encoded_row)
        return transformed

    def get_feature_names_out(self):
        """
            Returns output feature names after one-hot encoding
        """
        return [f"cat_{cat}" for cat in self.categories]


