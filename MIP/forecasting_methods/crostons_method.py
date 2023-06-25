import numpy as np
from matplotlib import pyplot as plt



import numpy as np
from matplotlib import pyplot as plt


class TSB:
    def __init__(self, alpha_d=0.2, alpha_p=0.2):
        self.alpha_d = alpha_d
        self.alpha_p = alpha_p
        self.p = None
        self.q = None

    def fit(self, data):
        demand = np.array(data)

        # Initialize first observation
        first_demand = np.argmax(demand > 0)
        self.p = demand[first_demand]
        self.q = 1.0

        # Calculate smoothed estimates for remaining observations
        for i, val in enumerate(demand[first_demand + 1:]):
            self.p = self.alpha_d * val + (1 - self.alpha_d) * self.p
            if val > 0:
                self.q = self.alpha_p * i + (1 - self.alpha_p) * self.q

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
        model = TSB(alpha_d=0.1, alpha_p=0.2)
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
        model = TSB(0.1, 0.1)
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
        print(f'MAE Croston´s method: {mae:.2f}')
        print(f'Std dev Croston´s method: {std_dev:.2f}')

    predictions = forecast.tolist()
    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    return mae, mse, rmse
