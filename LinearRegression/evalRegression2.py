from LinearRegression import LinearRegression
from sklearn.datasets import load_iris
iris = load_iris()

model = LinearRegression()
data = iris.data
X = data[:, [3, 1, 2]]
y = data[:, [0]]
model.load_weights("LinearRegression\modelWeights\Regression2.pkl")
print(model.score(X, y))
