import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

from MIP.analysis import analyse_data


def generate_data(num_ids: int, weeks: int = 204):
    """Generate a DataFrame with num_ids unique IDs and random transactions over specified weeks."""
    dates = pd.date_range(end=datetime.datetime.today(), periods=weeks, freq='W').to_pydatetime().tolist()
    data = {'ID': [], 'date': [], 'transaction_amount': []}

    # Create a seasonal pattern with a sine wave
    t = np.linspace(0, 4 * np.pi, weeks)
    seasonal_pattern = 5 + np.sin(t)

    for i in range(num_ids):
        data['ID'].extend([f'ID_{i}'] * weeks)
        data['date'].extend(dates)
        # Add seasonal pattern to random transaction amounts
        data['transaction_amount'].extend(np.random.uniform(0, 5, weeks) + seasonal_pattern)

    df = pd.DataFrame(data)
    df['date'] = df['date'].dt.date  # Convert to just date (no time)
    df.to_csv('transactions.csv', index=False)  # Save DataFrame to file
    return df


def plot_data(df):
    """Plot the transaction amounts over time."""
    df['date'] = pd.to_datetime(df['date'])  # Ensure date column is datetime type
    df_grouped = df.groupby(['date', 'ID']).sum().reset_index()  # Group by date and ID, summing transaction amounts

    plt.figure(figsize=(12, 6))
    for id in df_grouped['ID'].unique():
        plt.plot(df_grouped[df_grouped['ID'] == id]['date'], df_grouped[df_grouped['ID'] == id]['transaction_amount'], label=id)
    plt.xlabel('Date')
    plt.ylabel('Transaction Amount')
    plt.legend()
    plt.show()

def generate_seasonal_data_for_erratic_demand(products, num_periods, seed=3, period=52):
    products_list = []
    if seed is not None:
        np.random.seed(seed)

    for product in products:
        product = product.iloc[-208:]

        res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=period)
        num_repetitions = int(np.ceil(num_periods / len(product["sales_quantity"])))

        seasonal_filled = res.seasonal.fillna(method='ffill')

        seasonal_data = np.tile(seasonal_filled, num_repetitions)[:num_periods]
        data = seasonal_data

        # trend = np.linspace(product["sales_quantity"].mean(), product["sales_quantity"].mean(), num_periods)
        # data += trend

        residuals = res.resid.dropna()

        # Change the noise to use Poisson distribution.
        # We use the absolute value of the mean of residuals as lambda (rate parameter for the Poisson).
        lambda_ = abs(residuals.mean() + product["sales_quantity"].mean())
        if seed is None:
            rng = np.random.default_rng()
            noise = rng.poisson(lambda_, num_periods)
        else:
            noise = np.random.poisson(lambda_, num_periods)

        data += noise
        new_index = pd.date_range(product["sales_quantity"].index[0], periods=num_periods, freq='W')
        product = product.reindex(new_index)
        product["sales_quantity"] = pd.Series(data, index=new_index)
        product["sales_quantity"] = product["sales_quantity"].clip(lower=0)
        products_list.append(product)

    return products_list



def generate_seasonal_data_for_smooth_demand(products, num_periods, seed=None, period=52):
    products_list = []
    if seed is not None:
        np.random.seed(seed)

    for product in products:
        product = product.iloc[-208:]

        res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=period)
        num_repetitions = int(np.ceil(num_periods / len(product["sales_quantity"])))

        seasonal_filled = res.seasonal.fillna(method='ffill')

        seasonal_data = np.tile(seasonal_filled, num_repetitions)[:num_periods]
        data = seasonal_data

        residuals = res.resid.dropna()
        var_resid = residuals.var()

        p = product["sales_quantity"].mean() / var_resid
        if p <= 0 or p >= 1 or np.isnan(p):
            # print(f"Invalid p: {p}. Using default p=0.5")
            p = 0.5  # default value
        n = product["sales_quantity"].mean() ** 2 / (var_resid - product["sales_quantity"].mean())
        if np.isnan(n) or n <= 0:
            # print(f"Invalid n: {n}. Using default n=1")
            n = 1  # default value

        # # Generate noise using Negative Binomial distribution
        if seed is None:
            rng = np.random.default_rng()
            noise = rng.negative_binomial(n=n, p=p, size=num_periods)
        else:
            noise = np.random.negative_binomial(n=n, p=p, size=num_periods)

        data += noise
        new_index = pd.date_range(product["sales_quantity"].index[0], periods=num_periods, freq='W')
        product = product.reindex(new_index)
        product["sales_quantity"] = pd.Series(data, index=new_index)
        product["sales_quantity"] = product["sales_quantity"].clip(lower=0)
        products_list.append(product)

    return products_list

def generate_seasonal_data_for_intermittent_demand(products, num_periods, p_demand = 0.5, seed = None):
    products_list = []
    if seed is not None:
        np.random.seed(seed)

    for product in products:
        product = product.iloc[-208:]
        # Decompose the series to get the seasonal and trend component
        res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=52)
        num_repetitions = int(np.ceil(num_periods / len(product["sales_quantity"])))
        p_demand = np.mean(product["sales_quantity"]>0)

        # Fill NaN values in the trend and seasonal components
        trend_filled = res.trend.fillna(method='ffill').fillna(method='bfill')
        seasonal_filled = res.seasonal.fillna(method='ffill')

        seasonal_data = np.tile(seasonal_filled, num_repetitions)[:num_periods]
        data = seasonal_data

        # Kernel Density Estimation for residuals
        # Filter positive residuals first if you're modeling demand size in positive demand periods
        fit = product["sales_quantity"] - seasonal_filled
        fit = fit[fit > 0]  # only consider positive residuals

        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(fit.values.reshape(-1, 1))

        # Generate noise by resampling from the kernel density estimate
        noise = kde.sample(num_periods)

        # Generate demand occurrences using a Bernoulli process
        demand_occurrences = np.random.choice([0, 1], size=num_periods, p=[1 - p_demand, p_demand])

        # Combine demand occurrences and sizes to generate demand data
        data += np.where(demand_occurrences == 1, noise.ravel(), 0)
        data = np.round(data).astype(int)


        new_index = pd.date_range(product["sales_quantity"].index[0], periods=num_periods, freq='W')
        product = product.reindex(new_index)

        product["sales_quantity"] = pd.Series(data, index=new_index)
        product["sales_quantity"] = product["sales_quantity"].clip(lower=0)
        products_list.append(product)

    return products_list



def generate_seasonal_data_for_intermittent_demand(products, num_periods, p_demand = 0.5, seed = None):
    products_list = []
    if seed is not None:
        np.random.seed(seed)

    for product in products:
        product = product.iloc[-208:]
        # Decompose the series to get the seasonal and trend component
        res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=52)
        num_repetitions = int(np.ceil(num_periods / len(product["sales_quantity"])))
        p_demand = np.mean(product["sales_quantity"]>0)

        # Fill NaN values in the trend and seasonal components
        trend_filled = res.trend.fillna(method='ffill').fillna(method='bfill')
        seasonal_filled = res.seasonal.fillna(method='ffill')

        seasonal_data = np.tile(seasonal_filled, num_repetitions)[:num_periods]
        data = seasonal_data

        trend = np.tile(trend_filled, num_repetitions)[:num_periods]
        # data += trend

        # Kernel Density Estimation for residuals
        # Filter positive residuals first if you're modeling demand size in positive demand periods
        fit = product["sales_quantity"] - seasonal_filled
        pos_resid = res.resid[res.resid > 0].dropna()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(fit.values.reshape(-1, 1))

        # Generate noise by resampling from the kernel density estimate
        noise = kde.sample(num_periods)

        # Generate demand occurrences using a Bernoulli process
        demand_occurrences = np.random.choice([0, 1], size=num_periods, p=[1 - p_demand, p_demand])

        # Generate demand size using the Poisson distribution or other method
        # lambda_ = np.mean(res.resid[res.resid > 0].dropna()) + np.mean(res.trend[res.resid > 0].dropna())
        # demand_size = np.random.poisson(lam=lambda_, size=num_periods)  # Modify based on your desired distribution

        # Combine demand occurrences and sizes to generate demand data
        data += np.where(demand_occurrences == 1, noise.ravel(), 0)
        data = np.round(data).astype(int)


        new_index = pd.date_range(product["sales_quantity"].index[0], periods=num_periods, freq='W')
        product = product.reindex(new_index)

        product["sales_quantity"] = pd.Series(data, index=new_index)
        product["sales_quantity"] = product["sales_quantity"].clip(lower=0)
        products_list.append(product)

    return products_list


# Generate new data for each product and store them in a list
# new_data = [generate_seasonal_data(product, num_periods=100) for product in products]


def generate_next_week_demand(product_series):
    # Decompose the product_series into trend, seasonal, and residual components
    res = sm.tsa.seasonal_decompose(product_series, model='additive', period=52)

    # Get the current week number (assuming product_series is indexed by date)
    current_week = product_series.index[-1].week

    # If current_week is 52, next week is 1, otherwise it is current_week + 1
    next_week = 1 if current_week == 52 else current_week + 1

    # Predict next week's trend component
    trend_prediction = product_series.mean()

    # Predict next week's seasonal component
    seasonal_prediction = res.seasonal[next_week - 1]  # subtract 1 because Python uses 0-based indexing

    # Modify the standard deviation calculation to also depend on seasonality
    noise_std_dev = 0.1 * (trend_prediction + seasonal_prediction)

    noise = np.random.normal(0, noise_std_dev)
    # Generate next week's noise component
    # noise = np.random.normal(loc=res.resid.mean(), scale=res.resid.std(), size=1)

    # Add all components together to get the synthetic data for next week
    demand = trend_prediction + seasonal_prediction + noise

    return demand
