from statsforecast.models import TSB

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
        model = TSB(smoothing_level=alpha)  # use TSB model
        model.fit(train.values)

    forecast = model.predict(n_time_periods)
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
        model = TSB(smoothing_level=alpha)  # use TSB model
        model.fit(train.values)

    forecast = model.predict(n_time_periods)

    if shouldShowPlot:
        plt.plot(test.index[:n_time_periods], test.values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast, label='Forecast')

        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title(f"Sales Forecast for product '{product_hash}' using TSB method")
        plt.legend()
        plt.show()
    forecast[forecast < 0] = 0
    mse = np.mean((test.values[:n_time_periods] - forecast) ** 2)
    mae = np.mean(np.abs((test.values[:n_time_periods] - forecast)))
    std_dev = np.std(test.values[:n_time_periods] - forecast)  # Standard deviation of forecast errors
    rmse = np.sqrt(np.mean((test.values[:n_time_periods] - forecast) ** 2))

    if verbose:
        print(f'sMAPE TSB method: {smape:.2f}')
        print(f'MAE TSB method: {mae:.2f}')
        print(f'Std dev TSB method: {std_dev:.2f}')

    predictions = forecast.tolist()
    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    return mae, mse, rmse
