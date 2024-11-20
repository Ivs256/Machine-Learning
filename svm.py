import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
data = pl.read_excel('Raisin_Dataset.xlsx')
print(data.head())
X = data.select(['Eccentricity', 'ConvexArea'])
Y = data['Class']
X = X.to_numpy()
Y = Y.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
Y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print("Classification Report:\n", classification_report(Y_test, Y_pred))
# plt.plot(X_test[:, 0], X_test[:, 1])
# plt.plot(Y_train)
# #plt.plot(accuracy)
# plt.title('SVM Decision Boundary')
# plt.xlabel('Feature1')
# plt.ylabel('Feature2')
# plt.show()