from datetime import timedelta

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product


def optimize_SARIMA(df, start_date):
    """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    p = range(0, 3, 1)
    d = 1
    q = range(0, 3, 1)
    P = range(0, 3, 1)
    D = 1
    Q = range(0, 3, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)

    df = df.asfreq("W")
    # Getting the product hash
    product_hash = df["product_hash"].iloc[0]
    df = df["sales_quantity"]

    train = df.loc[df.index <= start_date]
    test = df.loc[df.index > start_date]

    results = []
    count = 0
    for param in tqdm_notebook(parameters_list):
        print(str(count) + "/" + str(len(tqdm_notebook(parameters_list))))
        count += 1
        try:
            model = SARIMAX(train, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)

    return result_df


def forecast(df, start_date, model = None, n_time_periods=20, order=(1, 0, 1), seasonal_order=(1, 0, 1, 52), verbose=False):
    # Set the frequency of the index to weekly
    df = df.asfreq("W")
    # Getting the product hash

    product_hash = df["product_hash"].iloc[0]

    train = df.loc[df.index <= start_date]
    train = train.resample('W').asfreq().fillna(0)
    train_df = train
    train = train["sales_quantity"]


    # Check if a model file already exists for the given product_hash
    model_filename = f"sarima_models/model_{product_hash}.pkl"
    if verbose:
        print("product_hash", model_filename)
    if os.path.exists(model_filename) and os.path.getsize(model_filename) > 0 and model is None:
        try:
            # Load the existing model
            model = pd.read_pickle(model_filename)
        except EOFError:
            print(f"EOFError: File {model_filename} is empty or does not exist.")
        if verbose:
            print("Using existing model for product hash: ", 1)
    if model is None:
        # Fit a new SARIMA model

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, initialization='approximate_diffuse')
        model = model.fit(maxiter=100)
        # Save the model
        # model.save(model_filename)
    else:
        # Update the existing model with the latest data
        model = model.update(train)

    # Forecast the next 20 periods
    end_date = start_date + pd.DateOffset(weeks=n_time_periods)
    forecast = model.get_prediction(start=start_date + timedelta(days=7), end=end_date, dynamic=False)
    predictions = forecast.predicted_mean.tolist()
    predictions = np.array(predictions)
    predictions[predictions < 0] = 0
    # Inserting 0 at time period 0
    predictions = np.insert(predictions, 0, 0)
    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    # Dividing by 10 because safety stock is too high
    return predictions, model, train_df


def forecast_analyse(df, start_date, model = None, n_time_periods=20, order=(1, 0, 1), seasonal_order=(1, 0, 1, 52), shouldShowPlot=False, verbose=False):
    # Set the frequency of the index to weekly
    df = df.asfreq("W")
    # Getting the product hash

    product_hash = df["product_hash"].iloc[0]

    train = df.loc[df.index <= start_date]
    train = train.resample('W').asfreq().fillna(0)
    train_df = train
    test = df.loc[df.index > start_date]
    test = test.resample('W').asfreq().fillna(0)
    train = train["sales_quantity"]
    test = test["sales_quantity"]


    # Check if a model file already exists for the given product_hash
    model_filename = f"sarima_models/model_{product_hash}.pkl"
    if verbose:
        print("product_hash", model_filename)
    if os.path.exists(model_filename) and os.path.getsize(model_filename) > 0 and model is None:
        try:
            # Load the existing model
            model = pd.read_pickle(model_filename)
        except EOFError:
            print(f"EOFError: File {model_filename} is empty or does not exist.")
        if verbose:
            print("Using existing model for product hash: ", 1)
    if model is None:
        # Fit a new SARIMA model

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, initialization='approximate_diffuse')
        model = model.fit(maxiter=100, disp=0)
        # Save the model
        # model.save(model_filename)
    else:
        # Update the existing model with the latest data
        model = model.update(train)

    # Forecast the next 20 periods
    end_date = start_date + pd.DateOffset(weeks=n_time_periods)
    forecast = model.get_prediction(start=start_date + timedelta(days=7), end=end_date, dynamic=False)

    # calculate standard deviation:
    conf_int = forecast.conf_int(alpha=0.05)  # 95% confidence interval

    if shouldShowPlot:
        # Evaluate the forecast
        # plt.plot(train.index, train.values, label='Actual')
        plt.plot(test.index[:n_time_periods], test.values[:n_time_periods], label='Actual')
        plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast')
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', label='95% Confidence Interval')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.title('Sales Forecast SARIMA method')
        plt.legend()

        # Show the plot
        plt.show()
    predictions = forecast.predicted_mean.tolist()
    predictions = np.array(predictions)
    predictions[predictions < 0] = 0

    mse = np.mean((test.values[:n_time_periods] - predictions) ** 2)
    mae = np.mean(np.abs(test.values[:n_time_periods] - predictions))
    std_dev = np.std(test.values[:n_time_periods] - predictions)  # Standard deviation of forecast errors
    rmse = np.sqrt(np.mean((test.values[:n_time_periods] - predictions) ** 2))
    smape = np.mean(200 * np.abs(test.values[:n_time_periods] - predictions) / (np.abs(test.values[:n_time_periods]) + np.abs(predictions)))
    cv = np.std(test.values[:n_time_periods] - predictions) / np.mean(test.values[:n_time_periods]) * 100


    if verbose:
        print(f'MSE SARIMA: {mse:.2f}')
        print(f'MAE SARIMA: {mae:.2f}')
        print(f'Std dev SARIMA: {std_dev:.2f}')


    if np.isnan(predictions).any():
        raise ValueError("predictions contains NaN values")

    # Dividing by 10 because safety stock is too high
    return mae, mse, rmse
