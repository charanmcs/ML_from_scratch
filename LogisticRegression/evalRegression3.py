from LogisticRegression import LogisticRegression
from sklearn.datasets import load_iris
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import numpy as np
iris = load_iris()
y=iris.target
X = iris.data[:, [2,3]]    #Taking in sepal length and width
model = LogisticRegression()
model.load_weights("LogisticRegression\modelWeights\TrainedWithPetalDimensions.pkl")
print(model.score(X, y))
fig = plt.figure(figsize=(10, 8))
fig=plot_decision_regions(X,np.array(y),model)
plt.savefig("LogisticRegression\modelPlots\plot3.png")
