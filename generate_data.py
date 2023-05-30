import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm


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

def generate_seasonal_data_based_on_products(products, num_periods, seed = None):
    products_list = []
    if seed is not None:
        np.random.seed(seed)
    for product_series in products:
        # First, we decompose the series to get the seasonal component
        res = sm.tsa.seasonal_decompose(product_series, model='additive', period=52)
        # Then, we repeat the seasonal component for the desired number of periods
        seasonal_data = np.tile(res.seasonal, num_periods // len(product_series) + 1)[:num_periods]
        data = seasonal_data

        # We add a trend and some noise to make it more realistic
        trend = np.linspace(product_series.mean(), product_series.mean(), num_periods)
        data += trend
        # Modify the standard deviation calculation to also depend on seasonality
        noise_std_dev = 0.1 * (trend + seasonal_data)
        noise = np.random.normal(0, noise_std_dev)
        # noise = np.random.normal(loc=res.resid.mean(), scale=res.resid.std(), size = num_periods)
        data += noise
        products_list.append(pd.Series(data, index=pd.date_range(product_series.index[0], periods=num_periods, freq='W')))

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



# # Generate and plot data
# df = generate_data(5, 30)
# plot_data(df)
