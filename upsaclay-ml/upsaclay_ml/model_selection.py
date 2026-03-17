# upsaclay-ml/upsaclay_ml/model_selection.py

import random

"""
A train/test split is a way to divide your dataset into two parts to avoid overfitting: 
one for training the model and one for testing it. 
The model learns from the training set, then we evaluate its performance on the test set to see how well it works on unseen data. 
"""
def train_test_split(X, y):
    """
    Splits X and y into training and test sets 
        X: Features 
        y: Labels

    Returns X_train, X_test, y_train, y_test after splitting
    """
    pairs = []
    # Making pairs of label/feature
    for i in range(len(X)): # len(X) = len(y)
        pairs.append((X[i], y[i]))
    random.shuffle(pairs) # Gambling ahh shuffle

    test_size = 0.2 # You can change the value
    test_count = int(len(X) * test_size) 
    # Getting the good index so we can split
    X_test = [pair[0] for pair in pairs][:test_count]
    y_test = [pair[1] for pair in pairs][:test_count]
    X_train = [pair[0] for pair in pairs][test_count:]
    y_train = [pair[1] for pair in pairs][test_count:]

    return X_train, X_test, y_train, y_test




