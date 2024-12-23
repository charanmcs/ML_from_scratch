import pickle
import numpy as np
import random

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Logistic Regression using Gradient Descent.

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
        self.num_classes = None

    def fit(
        self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, lr=0.01
    ):
        """Fit a logistic regression model.

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
        

        # Initialize weights and bias
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1], len(np.unique(y))))
        self.bias = np.random.uniform(-1, 1, size=(len(np.unique(y)),))
        self.num_classes = len(np.unique(y))

        # Convert y to one-hot encoding
        y_one_hot = np.eye(self.num_classes)[y]

        # Training loop
        for epoch in range(1, max_epochs):
            val_indices = random.sample(range(X.shape[0]), k=int(X.shape[0] * 0.1))
            X_val, y_val = X[val_indices], y_one_hot[val_indices]
            valid_loss = self.score(X_val, y_val)
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_true = y_one_hot[i : i + self.batch_size]
                if len(X_batch) < 1:
                    break
                y_pred = self.softmax(np.dot(X_batch, self.weights) + self.bias)
                w = (
                    -np.dot((y_true - y_pred).T, X_batch).T / X_batch.shape[0]
                    + self.regularization * self.weights
                )
                b = -np.mean(y_true - y_pred, axis=0)
                self.weights -= lr * w
                self.bias -= lr * b

            loss = self.score(X, y_one_hot)
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
        """Predict using the logistic regression model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        return np.argmax(self.softmax(np.dot(X, self.weights) + self.bias), axis=1)

    def score(self, X, y):
        """Evaluate the logistic regression model using cross-entropy loss.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        if not self.num_classes:
            self.num_classes = len(np.unique(y))
        try:
            response =  -np.mean(y * np.log(self.softmax(np.dot(X, self.weights) + self.bias)))
        except:
            y_one_hot = np.eye(self.num_classes)[y]
            response = -np.mean(y_one_hot * np.log(self.softmax(np.dot(X, self.weights) + self.bias)))
        return response


    def save(self, filePath):
        # Save weights and bias as pickle file
        with open(filePath, "wb") as f:
            pickle.dump((self.weights, self.bias), f)

    def load_weights(self, filePath):
        # Load weights and bias from pickle file
        with open(filePath, "rb") as f:
            self.weights, self.bias = pickle.load(f)

    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
