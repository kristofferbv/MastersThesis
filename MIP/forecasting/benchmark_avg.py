import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

def forecast_analysis(df, date, shouldShowPlot=False, verbose = False, n_time_periods=20):
    if isinstance(df, pd.DataFrame):
        # Split the data into training and testing sets
        train = df.loc[df.index <= date]
        test = df.loc[df.index > date]
    else:
        train = df.loc[df.index <= date]
        test = df.loc[df.index > date]

    # Create a 12-month rolling average from the training data
    rolling_avg = train["sales_quantity"].rolling(window=52).mean()

    # For the forecast, repeat the corresponding month's average value in the testing set
    forecast = pd.Series(index=test.index, dtype='float64')
    for i in range(n_time_periods):
        month = test.index[i].month
        forecast[test.index[i]] = rolling_avg[train.index.month == month].iloc[-1]

    # Ensure forecast is non-negative
    forecast[forecast < 0] = 0

    if shouldShowPlot:
        # Evaluate the forecast
        plt.plot(test.index[:n_time_periods], test["sales_quantity"].values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast.values[:n_time_periods], label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast 12-month average method')
        plt.legend()

        # Show the plot
        plt.show()

    mse = np.mean((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods]) ** 2)
    mae = np.mean(np.abs((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods])))
    rmse = np.sqrt(mse)

    if verbose:
        print(f'MAE 12-month average: {mae:.2f}')

    return mae, mse, rmse