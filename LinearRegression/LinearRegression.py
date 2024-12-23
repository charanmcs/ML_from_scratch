import pickle
import numpy as np
import random

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.num_outputs = None

    def fit(
        self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, lr=0.01
    ):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.lossi = []

        # Initializing the weights and bias based on the shape of X and y.
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1], y.shape[1]))
        self.bias = np.random.uniform(-1, 1, size=(y.shape[1],))
        self.num_outputs = y.shape[1]
        X = np.array(X)
        y = np.array(y)

        # training loop.
        for epoch in range(1, max_epochs):
            val_indices = random.sample(range(X.shape[0]), k=int(X.shape[0] * 0.1))
            X_val, y_val = X[val_indices], y[val_indices]
            valid_loss = self.score(X_val, y_val)
            self.lossi.append(valid_loss)
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_true = y[i : i + self.batch_size]
                if len(X_batch) < 1:
                    break
                y_pred = np.dot(X_batch, self.weights) + self.bias
                w = (
                    -(np.dot((y_true - y_pred).T, X_batch).T / X_batch.shape[0])
                    + self.regularization * self.weights
                )
                b = -np.mean(y_true - y_pred, axis=0)
                self.weights -= lr * w
                self.bias -= lr * b

            loss = self.score(X, y)
            if valid_loss < self.score(X_val, y_val):
                patience -= 1
            else:
                patience = self.patience
            if not patience:
                print("Exited as patience became zero")
                break
            print(
                f"At Epoch:{epoch} loss is {loss:.4f}, Validation loss is {valid_loss:.4f}"
            )

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        return np.mean((y - self.predict(X)) ** 2)

    def save(self, filePath):
        # Save weights and bias as pickle file
        with open(filePath, "wb") as f:
            pickle.dump((self.weights, self.bias), f)

    def load_weights(self, filePath):
        # Load weights and bias from pickle file
        with open(filePath, "rb") as f:
            self.weights, self.bias = pickle.load(f)
