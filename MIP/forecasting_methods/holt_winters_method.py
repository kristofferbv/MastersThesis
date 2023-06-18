# import pandas as pd
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

import statistics


def forecast(df, date, n_time_periods=20):
    if isinstance(df, pd.DataFrame):
        # Split the data into training and testing sets
        train = df.loc[df.index <= date]
    else:
        train = df.loc[df.index <= date]
    # train.index.freq = 'W-SUN'
    # train = train.asfreq('W-SUN')

    # Fit the model and make forecasts
    model = ExponentialSmoothing(train["sales_quantity"], seasonal_periods=52, trend='add', seasonal='add')

    fit = model.fit()
    # Forecasting len(test) periods ahead
    forecast = fit.forecast(n_time_periods)

    # making sure we don't forecast negative values
    forecast[forecast < 0] = 0

    return np.insert(forecast.values, 0, 0), train

def forecast_analysis(df, date, shouldShowPlot=False, verbose = False, n_time_periods=20, seasonal_periods = 52):
    print("season", seasonal_periods)
    if isinstance(df, pd.DataFrame):
        # Split the data into training and testing sets
        train = df.loc[df.index <= date]
        test = df.loc[df.index > date]
    else:
        train = df.loc[df.index <= date]
        test = df.loc[df.index > date]

    # train.index.freq = 'W-SUN'
    # train = train.asfreq('W-SUN')

    # Fit the model and make forecasts
    model = ExponentialSmoothing(train["sales_quantity"], seasonal_periods=seasonal_periods, trend='add', seasonal='add')

    fit = model.fit()
    # Forecasting len(test) periods ahead
    forecast = fit.forecast(n_time_periods)

    # making sure we don't forecast negative values
    forecast[forecast < 0] = 0
    mse = 0

    if shouldShowPlot:
        # Evaluate the forecast
        # plt.plot(train.index, train.values, label='Actual')
        plt.plot(test.index[:n_time_periods], test["sales_quantity"].values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast.values, label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast Holt-Winters method')
        plt.legend()

        # Show the plot
        plt.show()
    forecast[forecast < 0] = 0
    print(forecast)
    mse = np.mean((test["sales_quantity"].values[:n_time_periods] - forecast) ** 2)
    mae = np.mean(np.abs((test["sales_quantity"].values[:n_time_periods] - forecast)))
    std_dev = np.std(test["sales_quantity"].values[:n_time_periods] - forecast)  # Standard deviation of forecast errors
    rmse = np.sqrt(np.mean((test["sales_quantity"].values[:n_time_periods] - forecast) ** 2))

    if verbose:
        print(np.abs(test["sales_quantity"].values[:n_time_periods] - forecast))
        print(np.abs(test["sales_quantity"].values[:n_time_periods]) + np.abs(forecast))
        print(f'MAE Holt Winter: {mae:.2f}')
        print(f'Std dev Holt Winter: {std_dev:.2f}')

    try:
        standard_deviation = statistics.stdev((test["sales_quantity"].values[:n_time_periods] - forecast))
    except:
        print("df", df)
        print("date", date)

    return mae, mse, rmse

