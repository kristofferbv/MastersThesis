# import pandas as pd
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

import statistics


def forecast(df, date, shouldShowPlot=False, n_time_periods=20):
    if isinstance(df, pd.DataFrame):
        # Split the data into training and testing sets
        train = df.loc[df.index <= date]["sales_quantity"]
        test = df.loc[df.index > date]["sales_quantity"]
    else:
        train = df.loc[df.index <= date]
        test = df.loc[df.index > date]

    # train.index.freq = 'W-SUN'
    # train = train.asfreq('W-SUN')

    # Fit the model and make forecasts
    model = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='add')

    fit = model.fit()
    # Forecasting len(test) periods ahead
    forecast = fit.forecast(n_time_periods)

    # making sure we don't forecast negative values
    forecast[forecast < 0] = 0

    if shouldShowPlot:
        # Evaluate the forecast
        # plt.plot(train.index, train.values, label='Actual')
        plt.plot(test.index[:n_time_periods], test.values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast.values, label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast Holt-Winters method')
        plt.legend()

        # Show the plot
        plt.show()
        mse = np.mean((test.values[:n_time_periods] - forecast) ** 2)
        print(f'MSE: {mse:.2f}')

    standard_deviation = statistics.stdev((test.values[:n_time_periods] - forecast))

    return np.insert(forecast.values, 0, 0), [standard_deviation / 5] * (len(forecast) + 1)     # Dividing by 5 because safety stock is too high

