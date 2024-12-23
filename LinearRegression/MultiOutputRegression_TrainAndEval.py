import sys
import numpy as np
from LinearRegression import LinearRegression  # Assuming you have a custom implementation of LinearRegression
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Shuffle the data
np.random.shuffle(iris.data)

# Select features and target
X = iris.data[:, [0, 1]]    # Independent variables: 'sepal length (cm)', 'petal width (cm)'
y = iris.data[:, [2, 3]]    # Dependent variables: 'petal length (cm)', 'sepal width (cm)'

# Split the data into training and testing sets
n1 = int(X.shape[0] * 0.9)
X_train, y_train = X[:n1], y[:n1]
X_test, y_test = X[n1:], y[n1:]

# Train a linear regression model with multiple outputs and L2 regularization
model = LinearRegression()
model.fit(X_train, y_train, max_epochs=100, lr=0.001, regularization=0.1, patience=5)
print("Model score:", model.score(X_test, y_test))
