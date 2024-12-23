from LogisticRegression import LogisticRegression  # Importing the LogisticRegression class
import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
from sklearn.datasets import load_iris  # Importing the iris dataset from sklearn
from mlxtend.plotting import plot_decision_regions  # Importing plot_decision_regions for plotting decision regions
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
# Loading the iris dataset
iris = load_iris()
# Creating a DataFrame from the iris data
df = pd.DataFrame(iris.data)
# Adding the target variable to the DataFrame
df["target"] = iris.target
# Converting the DataFrame to a numpy array
data = df.values
# Shuffling the data randomly
np.random.shuffle(data)
# Extracting the target variable
y=data[:,[-1]]
# Converting the target variable to integers
y = [int(i) for i in y]
# Extracting the features
X = data[:, :-1]    #Taking all features in
# Splitting the data into training and testing sets
n1 = int(X.shape[0]*0.9)
X_train =X[:n1]
y_train = y[:n1]
X_test =X[:n1]
y_test = y[:n1]
# Creating an instance of the LogisticRegression model
model = LogisticRegression()
# Fitting the model to the training data with regularization=1 and patience=10
model.fit(X_train, y_train, regularization=1, patience=10)
# Printing the accuracy of the model on the test set
print(model.score(X_test,y_test))
# Saving the model weights to a file
model.save("LogisticRegression\modelWeights\AllfeaturesModel.pkl")
