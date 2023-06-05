import numpy as np
from matplotlib import pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import TSB



class Croston:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.p = None
        self.q = None

    def fit(self, data):
        demand = np.array(data)

        # Initialize first observation
        first_demand = np.argmax(demand > 0)
        self.p = demand[first_demand]
        self.q = 1.0

        # Calculate smoothed estimates for remaining observations
        for val in demand[first_demand + 1:]:
            if val > 0:
                self.p = self.alpha * val + (1 - self.alpha) * self.p
                self.q = self.alpha * 1 + (1 - self.alpha) * self.q

    def forecast(self, periods):
        return np.ones(periods) * (self.p / self.q)

def forecast(df, start_date, model=None, n_time_periods=20, alpha=0.1, shouldShowPlot=False, verbose=False):
    df = df.asfreq("W")
    product_hash = df["product_hash"].iloc[0]

    train = df.loc[df.index <= start_date]
    train = train.resample('W').asfreq().fillna(0)
    train_df = train
    test = df.loc[df.index > start_date]
    test = test.resample('W').asfreq().fillna(0)
    train = train["sales_quantity"]
    test = test["sales_quantity"]

    if model is None:
        model = Croston(alpha=alpha)
        model.fit(train)

    forecast = model.forecast(n_time_periods)
    forecast[forecast < 0] = 0
    predictions = forecast.tolist()
    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    return predictions, model, train_df

def forecast_analysis(df, start_date, model=None, n_time_periods=20, alpha=0.1, shouldShowPlot=False, verbose=False):
    df = df.asfreq("W")
    product_hash = df["product_hash"].iloc[0]

    train = df.loc[df.index <= start_date]
    train = train.resample('W').asfreq().fillna(0)
    train_df = train
    test = df.loc[df.index > start_date]
    test = test.resample('W').asfreq().fillna(0)
    train = train["sales_quantity"]
    test = test["sales_quantity"]

    if model is None:
        model = Croston(alpha=alpha)
        model.fit(train)

    forecast = model.forecast(n_time_periods)

    if shouldShowPlot:
        plt.plot(test.index[:n_time_periods], test.values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast, label='Forecast')

        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title(f"Sales Forecast for product '{product_hash}' using Croston\'s method")
        plt.legend()
        plt.show()
    forecast[forecast < 0] = 0
    mse = np.mean((test.values[:n_time_periods] - forecast) ** 2)
    mae = np.mean(np.abs((test.values[:n_time_periods] - forecast)))
    std_dev = np.std(test.values[:n_time_periods] - forecast)  # Standard deviation of forecast errors
    rmse = np.sqrt(np.mean((test.values[:n_time_periods] - forecast) ** 2))

    if verbose:
        print(f'sMAPE Croston´s method: {smape:.2f}')
        print(f'MAE Croston´s method: {mae:.2f}')
        print(f'Std dev Croston´s method: {std_dev:.2f}')

    predictions = forecast.tolist()
    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    return mae, mse, rmse
