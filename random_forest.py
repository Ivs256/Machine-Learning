import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
feature_importances = model.feature_importances_
print(f"Feature Importances: {feature_importances}")
# plt.barh(inputs_n.columns, feature_importances)
# plt.xlabel("Feature Importance")
# plt.title("Feature Importances in Random Forest")
# plt.show()
