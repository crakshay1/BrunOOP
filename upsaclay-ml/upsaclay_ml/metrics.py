# upsaclay-ml/upsaclay_ml/metrics.py

## SUPERVISED
def rand_score(X, Y, predicted_labels):
    """
        Returns the amount of point pairs correctly clustered according to true labels
        https://www.geeksforgeeks.org/machine-learning/rand-index-in-machine-learning/
        Basically, we try to get the pairs in common between true and predicted clustering
    """
    n = len(Y) # number of points in labels
    if n <= 1: # If we have 0 or 1 point, there are no pairs, score = 0
        return 0

    # Counters
    correct = 0  # number of pairs correctly classified
    total = 0    # total number of pairs examined

    for i in range(n):
        for j in range(i + 1, n): # i+1 Otherwise we are double-counting
            # Check if the pair i,j are in the same true class
            same_true = (Y[i] == Y[j])
            # Check if the pair i,j are in the same predicted cluster
            same_pred = (predicted_labels[i] == predicted_labels[j])
            
            # The pair is correct if both in same class & same predicted cluster
            # and both in different classes & different predicted clusters
            if same_true == same_pred:
                correct += 1
            total += 1

    # The amount of pairs correctly classified
    return correct / total


def adjusted_rand_score(X, Y, predicted_labels):
    """
        Returns the Adjusted Rand Index (ARI) between true labels and predicted labels
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
        ARI corrects the Rand Index : taking into account the expected number of correct pairs 
        if clusters were random and adjusts the score 
        by subtracting the expected value and normalizing
    """

    # We compute the number of pairs in a group of size k
    def count_pairs(k):
        return k * (k - 1) // 2

    n = len(Y)  # number of points in labels
    if n <= 1:       # If we have 0 or 1 point, there are no pairs, score = 0
        return 0  # Exactly like in rand_score

    # Counters
    contingency = {}  # counts of (true_label, predicted_label) pairs
    true_count = {}   # counts of each true label
    pred_count = {}   # counts of each predicted label

    for t, p in zip(Y, predicted_labels):
        contingency[(t, p)] = contingency.get((t, p), 0) + 1
        true_count[t] = true_count.get(t, 0) + 1
        pred_count[p] = pred_count.get(p, 0) + 1

    # Compute number of agreeing pairs
        # Agreeing pairs = the pairs in common between true and predicted clustering
    a = sum(count_pairs(v) for v in contingency.values()) 
        # same cluster & same true label
    b = sum(count_pairs(v) for v in true_count.values())    
        # total pairs in same true class
    c = sum(count_pairs(v) for v in pred_count.values())    
        # total pairs in same predicted cluster

    total_pairs = count_pairs(n) # total possible pairs
    expected = (b * c) / total_pairs # number of agreeing pairs we would expect by chance
    max_index = (b + c) / 2 # maximum possible number of agreeing pairs given the cluster sizes

    # Avoid division by zero if maximum = expected because we do max_index - expected
    if max_index == expected:
        return 0

    # Adjusted Rand Index = (observed - expected) / (max - expected)
    return (a - expected) / (max_index - expected)


# BROTATO IS NOT THE READER
# https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-v-measure
def homogeneity_score(X, Y, predicted_labels):
    """
        Measures how pure each predicted cluster is:
        - 1: all points in each predicted cluster come from the same true class
        - 0: clusters contain mixed true classes
    """
    clusters = {}  # group true labels by predicted cluster
    for t, p in zip(Y, predicted_labels):
        if p not in clusters:
            clusters[p] = []
        clusters[p].append(t)

    # Count clusters that contain points from only one true class
    pure = sum(1 for vals in clusters.values() if len(set(vals)) == 1)

    # Return fraction of clusters that are “pure”
    return pure / len(clusters)


def completeness_score(X, Y, predicted_labels):
    """
        Returns the Completeness Score of the clustering
        Measures how well all points of a given true class are assigned to the same predicted cluster:
        - 1 : all points of a class are in the same cluster
        - 0 : points of a class are split across multiple clusters
    """
    classes = {}  # dictionary to group points by true class
    for t, p in zip(Y, predicted_labels):
        classes.setdefault(t, []).append(p)  # append predicted labels to each true class

    # Count true classes that are fully contained within a single predicted cluster
    complete = sum(1 for vals in classes.values() if len(set(vals)) == 1)

    # Return fraction of classes that are “complete”
    return complete / len(classes)


def v_measure_score(X, Y, predicted_labels):
    """
        Returns the V-Measure of the clustering
        - 1 : perfect clustering (clusters are pure and complete)
        - 0 : poor clustering
    """
    h = homogeneity_score(X, Y, predicted_labels)   # homogeneity
    c = completeness_score(X, Y, predicted_labels)  # completeness

    # Avoid division by zero
    if h + c == 0:
        return 0
    
    return 2 * h * c / (h + c)


## UNSUPERVISED
def silhouette_score(X, labels):
    """
        Measures how similar each point is to its own cluster compared to other clusters:
        - 1 : well-clustered (far from other clusters and close to its own)
        - 0: overlapping clusters
        - -1 : bad-clustered
        Works with Euclidean distance
    """
    n = len(X)  # number of points
    if n <= 1:       # not enough points to form pairs
        return 0

    # Euclidean distance between two points
    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    scores = []
    for i in range(n):
        # Points in the same cluster 
        same = [j for j in range(n) if labels[j] == labels[i] and j != i]
        # Points in other clusters
        others = {}
        for l in set(labels):
            if l != labels[i]:  # skip the current point's cluster
                others[l] = [j for j in range(n) if labels[j] == l]

        if not same:   # cluster with one point: score = 0, no intra-cluster distance
            scores.append(0)
            continue
        
        # Notes be clutching 
        # Mean intra-cluster distance
        a = sum(dist(X[i], X[j]) for j in same) / len(same)
        # Minimum mean inter-cluster distance
        b = min(sum(dist(X[i], X[j]) for j in cluster) / len(cluster) 
                for cluster in others.values())
        # Silhouette score for this point
        scores.append((b - a) / max(a, b))

    # Average score over all points
    return sum(scores) / len(scores)


def mean_squared_error(Y, predicted_values):
    """
        Measures average squared difference between true and predicted values
        Lower MSE means that predictions are closer to true values
    """
    n = len(Y)
    if n == 0:  # avoid division by zero
        return 0
    return sum((y - p) ** 2 for y, p in zip(Y, predicted_values)) / n


def accuracy_score(Y, predicted_labels):
    """
        Measures fraction of correct predictions:
        - 1 : all predictions correct
        - 0 : no predictions correct
    """
    n = len(Y)
    if n == 0:  # avoid division by zero
        return 0
    correct = sum(1 for y, p in zip(Y, predicted_labels) if y == p)
    return correct / n

