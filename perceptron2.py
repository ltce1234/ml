import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

URL_='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data&#39;
df = pd.read_csv(URL_, header = None)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

print(X[0:5])
print(y[0:5])

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.25)

train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:5])

print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:5])

from sklearn.linear_model import Perceptron

perceptron = Perceptron(random_state = 42, max_iter = 20, tol = 0.001)
perceptron.fit(train_data, train_labels)


test_preds = perceptron.predict(test_data)
print(test_preds)

test_accuracy = accuracy_score(test_preds, test_labels)
print("Accuracy on test data: ", round(test_accuracy, 2) * 100, "%")