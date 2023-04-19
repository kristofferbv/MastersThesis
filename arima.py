import pandas as pd
import numpy as np
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt

def forecast(df, date):
    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]
    train.index.freq = 'W-SUN'

    # Fit the model and make forecasts
    model = auto_arima(train, seasonal=True, m=52, trace=True, error_action='ignore', suppress_warnings=True)
    forecast = model.predict(n_periods=len(test))

    # Evaluate the forecast
    plt.plot(test.index, test.values, label='Actual')
    plt.plot(forecast.index, forecast.values, label='Forecast')

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.title('Sales Forecast ARIMA')
    plt.legend()

    # Show the plot
    plt.show()

    # Evaluate the forecast
    mse = np.mean((test.values - forecast)**2)
    print(f'MSE: {mse:.2f}')
