from importlib.resources import files

from upsaclay_ml.pipeline import Pipeline
from upsaclay_ml.models.linear_model import LinearRegression
from upsaclay_ml.transformers import StandardScaler, OneHotEncoder
from upsaclay_ml.datasets import Dataset
from upsaclay_ml.metrics import mean_squared_error

def test_pipe():
    csv_path = files("upsaclay_ml").joinpath("test_dataset.csv")
    
    # Label is last column, header exists
    ds = Dataset(str(csv_path), has_header=True, target_col=-1)
    X, y = ds  # X: [id, age, gender, height_cm, weight_kg]
    
    GENDER_COL = 2
    
    # 1) One-hot encode gender (needs 2D n x 1)
    X_gender = [[row[GENDER_COL]] for row in X]
    enc = OneHotEncoder()
    X_gender_oh = enc.fit_transform(X_gender)
    
    # 2) Numeric part: drop id (col 0) and drop gender (col 2) numeric columns in X are: age (1), height (3), weight (4)
    X_num = [[row[1], row[3], row[4]] for row in X]
    
    # 3) Combine numeric + one-hot
    X_final = [xn + xoh for xn, xoh in zip(X_num, X_gender_oh)]
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    
    pipe.fit(X_final, y)
    y_pred = pipe(X_final)
    
    print("MSE:", mean_squared_error(y, y_pred))

if __name__ == "__main__":
    test_pipe()

