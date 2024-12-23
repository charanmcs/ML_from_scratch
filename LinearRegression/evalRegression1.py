from LinearRegression import LinearRegression
from sklearn.datasets import load_iris
iris = load_iris()

model = LinearRegression()
data = iris.data
X = data[:, [0, 1, 2]]
y = data[:, [3]]
model.load_weights("LinearRegression\modelWeights\Regression1.pkl")
print(model.score(X, y))
