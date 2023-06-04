import numpy as np
from sklearn.linear_model import LinearRegression

from config_utils import load_config

def get_initial_std_dev(df, n_time_periods):
    df = df.copy()
    # Calculate naive forecast
    df['naive_forecast'] = df['sales_quantity'].shift()

    # Drop the first row, as it has no forecast
    df = df.iloc[1:]

    # Calculate forecast errors
    df.loc[:, 'forecast_error'] = df['sales_quantity'] - df['naive_forecast']

    # Calculate average demand
    df.loc[:, 'average_demand'] = df['sales_quantity'].rolling(4).mean().dropna()
    df = df.dropna()

    # Calculate log of forecast errors and average demand
    df['log_error'] = np.log(df['forecast_error'].abs() + 1e-10)
    df['log_demand'] = np.log(df['average_demand'] + 1e-10)

    # check for zero or negative values in 'average_demand'
    assert (df['average_demand'] < 0).sum() == 0

    # check for NaN values in 'average_demand' and 'log_demand'
    assert df[['average_demand', 'log_demand']].isna().sum().sum() == 0

    # Remove rows with NaN (resulting from log of zero or negative)
    df = df.dropna()

    # Train a linear regression model
    model = LinearRegression()
    model.fit(df[['log_demand']], df['log_error'])

    # The model coefficients correspond to log k1 and k2
    log_k1 = model.intercept_
    k2 = model.coef_[0]

    # Calculate standard deviation for each row
    initial_std_dev = np.exp(log_k1) * df['average_demand'].iloc[-1] ** k2
    initial_std_devs = []
    for i in range(n_time_periods + 1):
        # Using Axsäters method to decide forecast errors over multiple time periods. See page 28 in inventory control
        initial_std_devs.append(initial_std_dev*np.sqrt(i))

    return initial_std_devs


def get_std_dev(prev_std_dev, forecast_error,  n_time_periods, alpha=0.1):
    # Convert standard deviation for time t-1 to MAD
    prev_mad = prev_std_dev / np.sqrt(np.pi / 2)

    # Compute MAD for time t
    current_mad = (1 - alpha) * prev_mad + alpha * abs(forecast_error)

    # Convert MAD at time t to standard deviation
    current_std_dev = np.sqrt(np.pi / 2) * current_mad
    current_std_devs = []
    for i in range(n_time_periods + 1):
        # Using Axsäters method to decide forecast errors over multiple time periods. See page 28 in inventory control
        current_std_devs.append(current_std_dev * np.sqrt(i))

    return current_std_devs
