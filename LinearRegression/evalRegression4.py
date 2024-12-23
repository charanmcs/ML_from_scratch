from LinearRegression import LinearRegression
from sklearn.datasets import load_iris
iris = load_iris()

model = LinearRegression()
data = iris.data
X = data[:, [0, 1, 3]]
y = data[:, [2]]
model.load_weights("LinearRegression\modelWeights\Regression4.pkl")
print(model.score(X, y))
