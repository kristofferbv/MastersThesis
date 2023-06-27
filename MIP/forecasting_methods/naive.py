
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

import matplotlib.font_manager

plt.rcParams["font.family"] = "CMU Concrete"
plt.rcParams["font.family"] = "CMU Concrete"
mpl.rc('font', family='CMU Concrete')

#


def forecast(df, date, n_time_periods=20):
    shifted_df = df["sales_quantity"].shift(52)

    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]


    forecast = shifted_df.loc[test.index]
    forecast[forecast < 0] = 0
    forecast = forecast.values[:n_time_periods]


    return np.insert(forecast, 0, 0), train


import statistics
def forecast_analysis(df, date, shouldShowPlot=False, verbose = False, n_time_periods=20):
    # Shift the 'sales_quantity' by 52 weeks
    shifted_df = df["sales_quantity"].shift(52)
    print("shifted_df")
    print(shifted_df)

    # Split the data into training and testing sets
    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]

    # For the forecast, take the 'sales_quantity' from the same period last year
    forecast = shifted_df.loc[test.index]
    print(forecast)
    print(date)

    # Ensure forecast is non-negative
    forecast[forecast < 0] = 0

    if shouldShowPlot:
        # Evaluate the forecast
        plt.plot(test.index[:n_time_periods], test["sales_quantity"].values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast.values[:n_time_periods], label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title('Sales Forecast using Naïve method')
        plt.legend()

        # Show the plot
        plt.show()

    mse = np.mean((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods]) ** 2)
    mae = np.mean(np.abs((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods])))
    rmse = np.sqrt(mse)

    if verbose:
        print(f'MAE for Naïve method: {mae:.2f}')

    return mae, mse, rmse
