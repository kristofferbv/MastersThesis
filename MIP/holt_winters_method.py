import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Load the data
def forecast(df, date, shouldShowPlot = False):
    # Split the data into training and testing sets
    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]

    # train.index.freq = 'W-SUN'
    # train = train.asfreq('W-SUN')

    # Fit the model and make forecasts
    model = ExponentialSmoothing(train, seasonal_periods=52, trend='add', seasonal='add')
    fit = model.fit()
    # Forecasting len(test) periods ahead
    forecast = fit.forecast(len(test))

    # making sure we don't forecast negative values
    forecast[forecast < 0] = 0


    if shouldShowPlot:
        # Evaluate the forecast
        plt.plot(train.index, train.values, label='Actual')
        plt.plot(test.index, test.values, label='Actual')
        plt.plot(test.index, forecast.values, label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast Holt-Winters method')
        plt.legend()

        # Show the plot
        plt.show()
        mse = np.mean((test.values - forecast)**2)
        # print(f'MSE: {mse:.2f}')
    return forecast.values

