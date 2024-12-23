from LogisticRegression import LogisticRegression
from sklearn.datasets import load_iris
iris = load_iris()

model = LogisticRegression()
model.load_weights("LogisticRegression\modelWeights\AllfeaturesModel.pkl")
print(model.score(iris.data, iris.target))

