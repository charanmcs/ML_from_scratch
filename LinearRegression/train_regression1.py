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
X = iris.data[:, [0, 1, 2]]  # Independent variables: 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)'
y = iris.data[:, [3]]        # Dependent variable: 'petal width (cm)'

# Split the data into training and testing sets
n1 = int(X.shape[0] * 0.9)
X_train, y_train = X[:n1], y[:n1]
X_test, y_test = X[n1:], y[n1:]

# Train a linear regression model without regularization
model = LinearRegression()
model.fit(X_train, y_train, max_epochs=100, lr=0.002, regularization=0, patience=5)
print("Model without regularization score:", model.score(X_test, y_test))

# Plot the loss over epochs for the model without regularization
plt.plot(model.lossi)
plt.savefig("LinearRegression\modelPlots\\trainRegression_1.png")
plt.clf()

# Train a linear regression model with L2 regularization
model_v2 = LinearRegression()
model_v2.fit(X_train, y_train, max_epochs=100, lr=0.002, regularization=0.1, patience=5)
print("Model with L2 regularization score:", model_v2.score(X_test, y_test))

# Plot the loss over epochs for the model with L2 regularization
plt.plot(model_v2.lossi)
plt.savefig("LinearRegression\modelPlots\\trainRegressionWithL2_1.png")

# Save the weights of the model with L2 regularization
model_v2.save("LinearRegression\modelWeights\Regression1.pkl")