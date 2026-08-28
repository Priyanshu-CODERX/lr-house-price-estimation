import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        """
        Initialize the Linear Regression model.

        Parameters
        ----------
        lr : float
            Learning rate for gradient descent.
        iterations : int
            Number of iterations to run gradient descent.
        """
        self.lr = lr
        self.iterations = iterations
        self.w = None  # Model weights (coefficients)
        self.b = None  # Bias (intercept)
        self.loss_history = []  # Stores cost function value per iteration

    def zscore_normalize_features(self, X, rtn_ms=False):
        """
        Apply z-score normalization to features.

        Formula: X_norm = (X - mu) / sigma

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Input features (m samples, n features).
        rtn_ms : bool, optional
            If True, return mean (mu) and standard deviation (sigma)
            along with normalized features.

        Returns
        -------
        X_norm : ndarray
            Normalized feature matrix.
        mu : ndarray
            Feature means (only if rtn_ms=True).
        sigma : ndarray
            Feature standard deviations (only if rtn_ms=True).
        """
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        X_norm = (X - mu) / sigma

        if rtn_ms:
            return (X_norm, mu, sigma)
        else:
            return X_norm

    def compute_cost(self, X, y, w, b):
        """
        Compute the Mean Squared Error (MSE) cost function.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Feature matrix.
        y : ndarray, shape (m,)
            Target vector.
        w : ndarray, shape (n,)
            Weight vector.
        b : float
            Bias term.

        Returns
        -------
        cost : float
            The value of the cost function.
        """
        m = X.shape[0]
        cost = 0.0

        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            cost += (f_wb_i - y[i]) ** 2
        cost = cost / (2 * m)

        return np.squeeze(cost)

    def compute_gradient(self, X, y, w, b):
        """
        Compute the gradients for weights and bias.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Feature matrix.
        y : ndarray, shape (m,)
            Target vector.
        w : ndarray, shape (n,)
            Current weight vector.
        b : float
            Current bias term.

        Returns
        -------
        dj_dw : ndarray, shape (n,)
            Gradient of cost with respect to weights.
        dj_db : float
            Gradient of cost with respect to bias.
        """
        m, n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):
                dj_dw[j] += err * X[i, j]
            dj_db += err

        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_dw, dj_db

    def fit(self, X, y):
        """
        Train the Linear Regression model using gradient descent.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Training features.
        y : ndarray, shape (m,)
            Training targets.
        """
        m, n = X.shape
        self.w = np.zeros(n)  # Initialize weights
        self.b = 0  # Initialize bias

        for _ in range(self.iterations):
            # Compute gradients
            dj_dw, dj_db = self.compute_gradient(X, y, self.w, self.b)
            # Update parameters
            self.w -= self.lr * dj_dw
            self.b -= self.lr * dj_db
            # Compute and store cost
            cost = self.compute_cost(X, y, self.w, self.b)
            self.loss_history.append(cost)

    def predict(self, X):
        """
        Predict target values for given input features.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Input features.

        Returns
        -------
        y_pred : ndarray, shape (m,)
            Predicted target values.
        """
        return np.dot(X, self.w) + self.b
