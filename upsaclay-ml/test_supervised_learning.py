from upsaclay_ml.models.tree import DecisionTreeClassifier  
from upsaclay_ml.datasets import Iris  
from upsaclay_ml.model_selection import train_test_split 
from upsaclay_ml.metrics import accuracy_score

def test_supervised():
  X, y = Iris()  
  X_train, X_test, y_train, y_test = train_test_split(X, y)  
  clf = DecisionTreeClassifier(max_depth=3)  
  clf.fit(X_train, y_train)  
  print("Accuracy:", accuracy_score(y_test, clf(X_test))) 

if __name__ == "__main__":
  test_supervised()
