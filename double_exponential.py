# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sn
# sn.set()
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# data = pd.Series([140, 150, 112, 132, 143, 121, 453, 160])
# alpha = 0.2
# beta = 0.2
# model = ExponentialSmoothing(data, trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta)
# forecast = model.forecast(2)
# print("Forecasted values:", forecast)
# plt.plot(data,color='green')
# plt.plot(model.fittedvalues ,color='orange')
# plt.plot(forecast,color='red')
# plt.show()
'''import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import Holt
data=pd.read_csv('sales_data.csv',encoding='unicode_escape')
sales = data['Sales']
alpha = 0.5
beta = 0.5
dt = Holt(sales)
model = dt.fit(smoothing_level=alpha, smoothing_slope=beta)
forecast = model.forecast(7)
plt.figure()
plt.plot(sales, label='Observed Data')
plt.plot(model.fittedvalues, label='Fitted values', color='orange')
plt.plot(range(len(sales), len(sales) + 7), forecast, label='Forecast', color='red')
print(forecast)
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()'''
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sn
sn.set()
data = pd.read_csv('sales_data.csv', encoding='unicode_escape')
sales = data['Sales']
alpha_values = np.linspace(0, 1)
beta_values = np.linspace(0, 1)
def des_forecast(sales, alpha, beta):
    level = sales.iloc[0]
    trend = sales.iloc[1] - sales.iloc[0]
    forecast = [level + trend]

    for t in range(1, len(sales)):
        previous_level = level
        level = alpha * sales.iloc[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        forecast.append(level + trend)

    return forecast
def find_best_alpha_beta(sales, alpha_values, beta_values):
    best_alpha = None
    best_beta = None
    min_mse = float("inf")
    best_forecast = []

    for alpha in alpha_values:
        for beta in beta_values:
            forecast = des_forecast(sales, alpha, beta)
            mse = mean_squared_error(sales, forecast)
            if mse < min_mse:
                min_mse = mse
                best_alpha = alpha
                best_beta = beta
                best_forecast = forecast

    return best_alpha, best_beta, best_forecast
def extend_forecast(sales, best_alpha, best_beta, forecast, future_steps=45):
    level = forecast[-1]
    trend = forecast[-1] - forecast[-2]

    for _ in range(future_steps):
        next_forecast = level + trend
        forecast.append(next_forecast)

        # Update level and trend
        previous_level = level
        level = best_alpha * sales.iloc[-1] + (1 - best_alpha) * (level + trend)
        trend = best_beta * (level - previous_level) + (1 - best_beta) * trend

    return forecast
best_alpha, best_beta, best_forecast = find_best_alpha_beta(sales, alpha_values, beta_values)
extended_forecast = extend_forecast(sales, best_alpha, best_beta, best_forecast)
for forecast_value in extended_forecast:
    print(f'{forecast_value:.2f}')
plt.plot(sales, label='Observed Data')
plt.plot(extended_forecast, label='Double Exponential Forecast', color='green')
plt.legend()
plt.show()'''