from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

import retrieve_data
from MIP import holt_winters_method, arima, recurrent_neural_network, sarima
# import holt_winters_method, arima, recurrent_neural_network
from config_utils import load_config
import simulation
# import deterministic_model
import os
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd

# import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)


def get_non_stationary_products(products, start_date=None, should_plot=True, verbose=False):
    n = len(products)

    # Loop through the dataframes and axes, plotting each dataframe on a separate axis
    column_name = 'column_name'  # Replace with the name of the column you want to plot
    alpha = 0.05
    stationary_products = []
    p_values = []
    for i, df in enumerate(products):
        ad_fuller_result = adfuller(df['sales_quantity'])
        if ad_fuller_result[1] > alpha:
            stationary_products.append(df)
            p_values.append(ad_fuller_result[1])
            if verbose:
                print("Non-stationarity is discovered!")
                print(f'ADF Statistic: {ad_fuller_result[0]}')
                print(f'p-value: {ad_fuller_result[1]}')
                print("product_hash", df["product_hash"].iloc[0])
    # Plotting it:
    if should_plot:
        fig, axes = plt.subplots(nrows=len(stationary_products), sharex=True, figsize=(10, len(stationary_products) * 3))
        for i, (ax, df) in enumerate(zip(axes, stationary_products)):
            if start_date is not None:
                print("KUUUUK")
                df = df.loc[df.index >= start_date]["sales_quantity"]
            df["sales_quantity"].plot(ax=ax, label=df["product_hash"].iloc[0] + " p-value: " + str(round(p_values[i], 3)))
            ax.set_ylabel('sales quantity')
            ax.legend()
            # Customize the plot by adding a title and x-axis label
        axes[0].set_title('Non-stationary sales quantity')
        axes[-1].set_xlabel('Date')
        # Show the plot
        plt.show()
    return stationary_products


def plot_sales_quantity(products, start_date=None):
    n = len(products)
    fig, axes = plt.subplots(nrows=n, sharex=True, figsize=(10, n * 3))
    print(len(products))
    print(len(axes))

    for i, (df, ax) in enumerate(zip(products, axes)):
        if start_date != None:
            df = df.loc[df.index >= start_date]["sales_quantity"]
        # Plotting it:
        df["sales_quantity"].plot(ax=ax, label=df["product_hash"].iloc[0])
        ax.set_ylabel('sales quantity')
        ax.legend()

    # Customize the plot by adding a title and x-axis label
    axes[0].set_title(f'Comparison of sales quantities for different products')
    axes[-1].set_xlabel('Date')

    # Show the plot
    plt.show()


def decompose_sales_quantity(df):
    freq = 12
    decomposition = seasonal_decompose(df["sales_quantity"], model='additive', period=freq)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Plot the original data
    df["sales_quantity"].plot(ax=ax1)
    ax1.set_title('Original Sales Data')

    # Plot the trend component
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend Component')

    # Plot the seasonal component
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal Component')

    # Plot the residuals
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    config = load_config("config.yml")
    n_time_periods = config["n_time_periods"]  # number of time periods

    # retrieve_data.categorize_products("data/sales_orders.csv", "m", True)
    # products = retrieve_data.read_products("2016-01-01", "2020-12-30", "w")
    products = retrieve_data.read_products_3("2016-01-01", "2020-12-30")
    print("number of products: ", len(products))
    # Reading the products created by the "read_products" function above
    # products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
    start_date = products[0].index[48]

    # products2 = retrieve_data.read_products_2("2016-01-01", "2020-12-30")
    # holt_winters_method.forecast(products[0]["sales_quantity"], start_date, shouldShowPlot=True)

    p = range(0, 3, 1)
    d = 1
    q = range(0, 3, 1)
    P = range(0, 3, 1)
    D = 1
    Q = range(0, 3, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    df = products[0]
    # plot_sales_quantity(products[:20])
    get_non_stationary_products(products)
    # decompose_sales_quantity(df)

    # sarima.optimize_SARIMA(parameters_list, 1, 1, 52, products[0], start_date)
    # sarima.forecast(products[0][["product_hash","sales_quantity"]], start_date)
    # recurrent_neural_network.forecast(products[0]["sales_quantity"], start_date)
    # print(start_date)
    # simulation.simulate(start_date, n_time_periods, products)

    """
    Algorithm for simulation optimization: 
        1) Find n products and use historical data to forecast the next k periods at time t = t0
        2) Find current inventory level for each product
        2) Once the forecast and start inventory level is decided, run the optimization algorithm to find the best option
            at time t = t0. 
        3) Store this option! 
        
        4) Now at time t = t0 + 1, use the new available data to create a forecast of the next k periods at time t = t0 + 1
        5) repeat point 2-4 until t = t0 + k
    """
