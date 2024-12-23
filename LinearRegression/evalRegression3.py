from LinearRegression import LinearRegression
from sklearn.datasets import load_iris
iris = load_iris()

model = LinearRegression()
data = iris.data
X = data[:, [0, 3, 2]]
y = data[:, [1]]
model.load_weights("LinearRegression\modelWeights\Regression3.pkl")
print(model.score(X, y))
