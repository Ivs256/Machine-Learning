from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
data = load_iris()
print(data)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confuse_matrix=confusion_matrix(y_test,y_pred)
print("Accuracy of the Naive Bayes model:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confuse_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(confuse_matrix, annot=True, fmt='d', cmap='Blues',)
plt.title("Confusion Matrix")
plt.xlabel("True Labels")
plt.ylabel("Predicted Labels")
plt.show()