import os
import sys
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append(os.path.dirname(__file__))

from linear_regression import LinearRegression
from data_utils import load_housing_data, prepare_features, train_test_split
from metrics import mean_squared_error, r2_score


def main():
    data = load_housing_data("./data/raw/housing.csv")

    features = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    target = "MEDV"

    X, y = prepare_features(data, features, target)

    model = LinearRegression(lr=0.01, iterations=1000)

    X_norm, mu, sigma = model.zscore_normalize_features(X, rtn_ms=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, seed=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    plt.plot(model.loss_history)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Training Loss Curve")
    plt.show()

    model_artifacts = {
        "weights": model.w,
        "bias": model.b,
        "mu": mu,
        "sigma": sigma,
        "features": features,
    }
    with open("./models/linear_regression_model.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)

    print("Model saved to ../models/linear_regression_model.pkl")


if __name__ == "__main__":
    main()
