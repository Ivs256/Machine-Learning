import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
data = pd.read_csv('diabetes.csv')
X = data.iloc[:, :-1].values  # Features
Y = data.iloc[:, -1].values
k = 5
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
def knn_classification(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [y_train[i] for i in k_indices]
        most_common_label = Counter(k_labels).most_common(1)[0][0]

        predictions.append(most_common_label)
    return predictions
y_pred = knn_classification(X_train, y_train, X_test, k)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")
plt.plot([accuracy] * len(X_test), label='Accuracy', color='green')
plt.title(f'KNN Classifier Accuracy: {accuracy:.2f}%')
plt.xlabel('Test Samples')
plt.ylabel('Accuracy (%)')
plt.show()





