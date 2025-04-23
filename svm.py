from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

modelsvm = SVC(kernel='linear')
modelsvm.fit(X, y)

predictions = modelsvm.predict(X)
accuracy = modelsvm.score(X, y)
print("Accuracy of SVM:", accuracy)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
plt.show()

new_samples = np.array([[0, 0], [4, 4]])
predictions = modelsvm.predict(new_samples)
print(predictions)
print("hello")

clf = SVC(kernel='linear')
clf.fit(X, y)
print(clf.predict(new_samples))
