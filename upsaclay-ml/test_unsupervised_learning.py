from upsaclay_ml.models.cluster import KMeans  
from upsaclay_ml.datasets import Blobs  
from upsaclay_ml.metrics import silhouette_score  

def test_unsupervised():
  X = Blobs(n_samples=200, centers=3)  
  kmeans = KMeans(n_clusters=3) 
  labels = kmeans.fit_predict(X)  
  print("Silhouette:", silhouette_score(X, labels))

if __name__ == "__main__":
  test_unsupervised()
