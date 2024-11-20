'''import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data = pd.read_csv("salaries.csv")
inputs = data.drop('salary_more_then_100k', axis='columns')
target = data['salary_more_then_100k']
company_1 = LabelEncoder()
job_1 = LabelEncoder()
degree_1 = LabelEncoder()
inputs['company_n'] = company_1.fit_transform(inputs['company'])
inputs['job_n'] = job_1.fit_transform(inputs['job'])
inputs['degree_n'] = degree_1.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company', 'job', 'degree'], axis="columns")
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
print(model.score(inputs_n, target))
plt.figure(figsize=(12, 8))  # Optional: adjust the size of the tree plot
tree.plot_tree(model, filled=True, feature_names=inputs_n.columns, class_names=["<=100k", ">100k"], rounded=True)
plt.show()'''
'''import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
data = pd.read_csv("weather_forecast_data.csv")
inputs = data.drop('Rain', axis='columns')
target = data['Rain']
Temperature_new = LabelEncoder()
Humidity_new = LabelEncoder()
Wind_Speed_new = LabelEncoder()
Cloud_Cover_new = LabelEncoder()
Pressure_new = LabelEncoder()
inputs['Temperature_new'] = Temperature_new.fit_transform(inputs['Temperature'])
inputs['Humidity_new'] = Humidity_new.fit_transform(inputs['Humidity'])
inputs['Wind_Speed_new'] = Wind_Speed_new.fit_transform(inputs['Wind_Speed'])
inputs['Cloud_Cover_new'] = Cloud_Cover_new.fit_transform(inputs['Cloud_Cover'])
inputs['Pressure_new'] = Pressure_new.fit_transform(inputs['Pressure'])
inputs_n = inputs.drop(['Temperature', 'Humidity', 'Wind_Speed','Cloud_Cover','Pressure'], axis="columns")
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
print(model.score(inputs_n, target))
plt.figure(figsize=(12, 8))  # Optional: adjust the size of the tree plot
tree.plot_tree(model, filled=True, feature_names=inputs_n.columns, class_names=["rain", "no rain"], rounded=True)
plt.show()'''
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv("weather_forecast_data.csv")
inputs = data.drop('Rain', axis='columns')
target = data['Rain']
Temperature_new = LabelEncoder()
Humidity_new = LabelEncoder()
Wind_Speed_new = LabelEncoder()
Cloud_Cover_new = LabelEncoder()
Pressure_new = LabelEncoder()
inputs['Temperature_new'] = Temperature_new.fit_transform(inputs['Temperature'])
inputs['Humidity_new'] = Humidity_new.fit_transform(inputs['Humidity'])
inputs['Wind_Speed_new'] = Wind_Speed_new.fit_transform(inputs['Wind_Speed'])
inputs['Cloud_Cover_new'] = Cloud_Cover_new.fit_transform(inputs['Cloud_Cover'])
inputs['Pressure_new'] = Pressure_new.fit_transform(inputs['Pressure'])
inputs_n = inputs.drop(['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure'], axis="columns")
X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
tree.plot_tree(model, filled=True, feature_names=X_train.columns, class_names=["no rain", "rain"], rounded=True)
plt.show()