# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sn
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# sn.set()
# data = pd.Series([140, 150, 112, 132, 143, 121, 453, 160])
# alpha = 0.2
# beta = 0.2
# gamma=0.2
# model = ExponentialSmoothing(data, trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta,smoothing_seasonal=gamma)
# forecast = model.forecast(2)
# print("Forecasted values:", forecast)
# plt.plot(data,color='green')
# plt.plot(model.fittedvalues,color='orange')
# plt.plot(forecast,color='red')
# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
import codecs
with codecs.open('sales_data.csv') as f:
    data = pd.read_csv(f)
sales = data['Sales']
alpha = 0.5
beta = 0.5
gamma = 0.5
seasonal_periods = 12
test = ExponentialSmoothing(sales, seasonal='add', seasonal_periods=seasonal_periods)
model = test.fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
forecast = model.forecast(7)
plt.plot(sales, label='Observed Data')
plt.plot(model.fittedvalues, label='Fitted values', color='orange')
plt.plot(range(len(sales), len(sales) + 7), forecast, label='Forecast', color='red')
print(forecast)
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()
'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sn
sn.set()
data = pd.read_csv('sales_data.csv', encoding='unicode_escape')
sales = data['Sales']
alpha_values = np.linspace(0, 1, 100)
beta_values = np.linspace(0, 1, 100)
gamma_values = np.linspace(0, 1, 100)
season_length = 12  # Assume a monthly seasonal cycle

def tes_forecast(sales, alpha, beta, gamma, season_length):
    level = sales.iloc[0]
    trend = sales.iloc[1] - sales.iloc[0]
    seasonals = [sales[i] / level for i in range(season_length)]
    forecast = [level + trend + seasonals[0]]

    for t in range(1, len(sales)):
        season_index = t % season_length
        previous_level = level
        level = alpha * (sales.iloc[t] / seasonals[season_index]) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        seasonals[season_index] = gamma * (sales.iloc[t] / level) + (1 - gamma) * seasonals[season_index]
        forecast.append(level + trend + seasonals[season_index])

    return forecast
def find_best_alpha_beta_gamma(sales, alpha_values, beta_values, gamma_values, season_length):
    best_alpha, best_beta, best_gamma = None, None, None
    min_mse = float("inf")
    best_forecast = []

    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                forecast = tes_forecast(sales, alpha, beta, gamma, season_length)
                mse = mean_squared_error(sales, forecast)
                if mse < min_mse:
                    min_mse = mse
                    best_alpha, best_beta, best_gamma = alpha, beta, gamma
                    best_forecast = forecast

    return best_alpha, best_beta, best_gamma, best_forecast
# Extend forecast for future steps
def extend_forecast(sales, best_alpha, best_beta, best_gamma, forecast, season_length, future_steps=45):
    level = forecast[-1] - forecast[-2]
    trend = forecast[-1] - forecast[-2]
    seasonals = [sales[i] / level for i in range(season_length)]

    for t in range(future_steps):
        season_index = (len(sales) + t) % season_length
        next_forecast = level + trend + seasonals[season_index]
        forecast.append(next_forecast)

        # Update level, trend, and seasonality
        previous_level = level
        level = best_alpha * (sales.iloc[-1] / seasonals[season_index]) + (1 - best_alpha) * (level + trend)
        trend = best_beta * (level - previous_level) + (1 - best_beta) * trend
        seasonals[season_index] = best_gamma * (sales.iloc[-1] / level) + (1 - best_gamma) * seasonals[season_index]

    return forecast
best_alpha, best_beta, best_gamma, best_forecast = find_best_alpha_beta_gamma(sales, alpha_values, beta_values,
                                                                              gamma_values, season_length)
extended_forecast = extend_forecast(sales, best_alpha, best_beta, best_gamma, best_forecast, season_length)
for forecast_value in extended_forecast:
    print(f'{forecast_value:.2f}')
plt.plot(sales, label='Observed Data')
plt.plot(extended_forecast, label='Triple Exponential Forecast', color='red')
plt.legend()
plt.show()'''