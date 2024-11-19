# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# import seaborn as sn
# sn.set()
# data = pd.Series([140,150,112,132,143,121,453,160])
# alpha=0.2
# model = SimpleExpSmoothing(data).fit(smoothing_level=alpha, optimized=False)
# forecast = model.forecast(2)
# print("Forecasted values:", forecast)
# plt.plot(data,color='green')
# plt.plot(model.fittedvalues ,color='orange')
# plt.plot(forecast,color='red')
# plt.show()
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import codecs

# Load the data
with codecs.open('sales_data.csv') as f:
    data = pd.read_csv(f)
sales = data['Sales']

# Set up alpha values and variables for finding the best alpha
alpha_values = np.linspace(0, 1, 100)
best_alpha = None
min_mse = float("inf")

# Loop through each alpha, apply SES manually, and calculate MSE
for alpha in alpha_values:
    # Initialize forecast array with the same length as sales
    forecast = [sales[0]]  # Initial forecast starts with the first observed value

    # Compute the forecast values for each time step
    for t in range(1, len(sales)):
        forecast.append(alpha * sales[t - 1] + (1 - alpha) * forecast[t - 1])

    # Calculate MSE for the current alpha
    mse_ses = mean_squared_error(sales, forecast)

    # Check if this MSE is lower than the current minimum MSE
    if mse_ses < min_mse:
        min_mse = mse_ses
        best_alpha = alpha
        best_forecast = forecast  # Store the best forecast so far

# Generate future forecast values for the next 45 time periods
forecast = list(best_forecast)
for _ in range(45):
    next_forecast = best_alpha * sales.iloc[-1] + (1 - best_alpha) * forecast[-1]
    forecast.append(next_forecast)
print(forecast)
data.to_csv("forecast.csv")
# Plot the observed data, fitted values, and the forecast
# plt.figure(figsize=(10, 6))
# plt.plot(sales, label='Observed Data')
# plt.plot(best_forecast, label='Fitted Values', color='orange')
# plt.plot(range(len(sales), len(sales) + 45), forecast_extended[-45:], label='Forecast', lw=2, color='red')
# plt.xlabel('Time')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sn
sn.set()
data=pd.read_csv('sales_data.csv',encoding='unicode_escape')
sales=data['Sales']
alpha_values = np.linspace(0, 1)
def ses_forecast(sales, alpha):
    forecast = [sales.iloc[0]]
    for t in range(1, len(sales)):
        forecast.append(alpha * sales.iloc[t - 1] + (1 - alpha) * forecast[t - 1])
    return forecast
def find_best_alpha(sales, alpha_values):
    best_alpha = None
    min_mse = float("inf")
    best_forecast = []
    for alpha in alpha_values:
        forecast = ses_forecast(sales, alpha)
        mse = mean_squared_error(sales, forecast)
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha
            best_forecast = forecast

    return best_alpha, best_forecast

def extend_forecast(sales, best_alpha, forecast, future_steps=45):
    for _ in range(future_steps):
        next_forecast = best_alpha * sales.iloc[-1] + (1 - best_alpha) * forecast[-1]
        forecast.append(next_forecast)
    return forecast
best_alpha, best_forecast = find_best_alpha(sales, alpha_values)
extended_forecast = extend_forecast(sales, best_alpha, best_forecast)
for forecast_value in extended_forecast:
    print(f'{forecast_value:.2f}')
plt.plot(sales, label='Observed Data')
plt.plot(extended_forecast,label='forecast',color='red')
plt.show()
