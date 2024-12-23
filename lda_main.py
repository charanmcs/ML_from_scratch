from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score
from models import LDAModel

train_data, train_labels, test_data, test_labels = load_and_prepare_data()

train_data = train_data.reshape(len(train_data), -1)
test_data = test_data.reshape(len(test_data), -1)

model = LDAModel()
model.fit(train_data,train_labels)
test_preds = model.predict(test_data)
print("RGB Score:",accuracy_score(test_labels, test_preds))


train_data, train_labels, test_data, test_labels = load_and_prepare_data(as_grayscale=True)
train_data = train_data.reshape(len(train_data), -1)
test_data = test_data.reshape(len(test_data), -1)

model = LDAModel()
model.fit(train_data,train_labels)
test_preds = model.predict(test_data)
print("Grey Scale Score:",accuracy_score(test_labels, test_preds))
