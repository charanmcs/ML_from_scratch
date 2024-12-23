import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from src.LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
iris = load_iris()
X = iris.data[:, [0,1]]
y = iris.target
model = LogisticRegression()
model.fit(X, y)
print(model.predict([X[58]]),y[58])
fig = plt.figure(figsize=(10, 8))
fig=plot_decision_regions(X,y,model)
plt.show()
sys.exit()





from src.LinearRegression import LinearRegression
# Independent -> 'sepal length (cm)', 'sepal width (cm)'
#  Dependent ->  'petal length (cm)', 'petal width (cm)'
X = iris.data[:, [0, 1]]
y = iris.data[:, [2, 3]]

model = LinearRegression()
model.fit(X, y)

from src.LinearRegression import LinearRegression
from sklearn.datasets import load_iris
iris = load_iris()





n1 = int(X.shape[0] * 0.9)
X_train =X[:n1]
y_train = y[:n1]
X_test =X[:n1]
y_test = y[:n1]
model = LinearRegression()
model.fit(X_train, y_train, max_epochs=100, lr=0.001, regularization=0.1, patience=5)
print(model.score(X_test, y_test))
print(model.predict(X[0]), y[0])




n1 = int(X.shape[0] * 0.9)
X_train =X[:n1]
y_train = y[:n1]
X_test =X[:n1]
y_test = y[:n1]
model = LinearRegression()
model.fit(X_train, y_train, max_epochs=100, lr=0.001, regularization=0.1, patience=5)
print(model.score(X_test, y_test))
print(model.predict(X[0]), y[0])



# Independent -> 'sepal length (cm)', 'sepal width (cm)', 'petal width (cm)'
#  Dependent ->  'petal length (cm)'
X = iris.data[:, [0, 1, 3]]
y = iris.data[:, [2]]
n1 = int(X.shape[0] * 0.9)
X_train =X[:n1]
y_train = y[:n1]
X_test =X[:n1]
y_test = y[:n1]
model = LinearRegression()
model.fit(X_train, y_train, max_epochs=100, lr=0.001, regularization=0.1, patience=5)
print(model.score(X_test, y_test))
print(model.predict(X[0]), y[0])