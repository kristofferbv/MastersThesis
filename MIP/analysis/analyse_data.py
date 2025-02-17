import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def get_non_stationary_products(products, start_date=None, should_plot=False, verbose=False):
    n = len(products)
    if isinstance(products[0], pd.DataFrame):
        is_dataframe = True
    else:
        is_dataframe = False

    # Loop through the dataframes and axes, plotting each dataframe on a separate axis
    column_name = 'column_name'  # Replace with the name of the column you want to plot
    alpha = 0.05
    stationary_products = []
    p_values = []
    for i, df in enumerate(products):
        if is_dataframe:
            label_name = df["product_hash"].iloc[0]
            ad_fuller_result = adfuller(df['sales_quantity'])

        else:
            label_name = str(i)
            ad_fuller_result = adfuller(df)
        if ad_fuller_result[1] > alpha:
            stationary_products.append(df)
            p_values.append(ad_fuller_result[1])
            if verbose:
                print("Non-stationarity is discovered!")
                print(f'ADF Statistic: {ad_fuller_result[0]}')
                print(f'p-value: {ad_fuller_result[1]}')
                print("product: ", label_name)
    # Plotting it:
    if should_plot and len(stationary_products) != 0:
        fig, axes = plt.subplots(nrows=len(stationary_products), sharex=True, figsize=(10, len(stationary_products) * 3))
        for i, (ax, df) in enumerate(zip(axes, stationary_products)):
            if is_dataframe:
                label_name = df["product_hash"].iloc[0]
                if start_date is not None:
                    df = df.loc[df.index >= start_date]["sales_quantity"]
                df["sales_quantity"].plot(ax=ax, label=label_name + " p-value: " + str(round(p_values[i], 3)))

            else:
                label_name = str(i)
                if start_date is not None:
                    df = df.loc[df.index >= start_date]
                df.plot(ax=ax, label=label_name + " p-value: " + str(round(p_values[i], 3)))

            ax.set_ylabel('sales quantity')
            ax.legend()
            # Customize the plot by adding a title and x-axis label
        axes[0].set_title('Non-stationary sales quantity')
        axes[-1].set_xlabel('Date')
        # Show the plot
        #plt.show()
    return stationary_products


def plot_sales_quantity(products, start_date=None):
    n = len(products)
    fig, axes = plt.subplots(nrows=n, sharex=True, figsize=(10, n * 3))
    if isinstance(products[0], pd.DataFrame):
        is_dataframe = True
    else:
        is_dataframe = False

    for i, (df, ax) in enumerate(zip(products, axes)):
        print("df", df)
        if is_dataframe:
            label_name = df["product_hash"].iloc[0]
            if start_date != None:
                df = df.loc[df.index >= start_date]["sales_quantity"]
                # Plotting it:
            df["sales_quantity"].plot(ax=ax, label=label_name)
        else:
            if start_date != None:
                df = df.loc[df.index >= start_date]["sales_quantity"]
            label_name = str(i)
            df.plot(ax=ax, label=label_name)

        ax.set_ylabel('sales quantity')
        ax.legend()

    # Customize the plot by adding a title and x-axis label
    axes[0].set_title(f'Comparison of sales quantities for different products')
    axes[-1].set_xlabel('Date')

    # Show the plot
    plt.show()


def decompose_sales_quantity(df, product_name = 0, freq = 52):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    if isinstance(df, pd.DataFrame):
        is_dataframe = True
        decomposition = seasonal_decompose(df["sales_quantity"], model='additive', period=freq)
        # Plot the original data
        df["sales_quantity"].plot(ax=ax1)
    else:
        is_dataframe = False
        decomposition = seasonal_decompose(df, model='additive', period=freq)
        df.plot(ax=ax1)

    ax1.set_title('Original Sales Data')

    # Plot the trend component
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend Component')

    # Plot the seasonal component
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal Component')

    # Plot the residuals
    decomposition.resid.plot(ax=ax4)
    _, p_value = stats.shapiro(decomposition.resid)
    if p_value < 0.05:
        print('Data do not follow a normal distribution')
    else:
        print('Data follow a normal distribution')
    ax4.set_title('Residuals')
    if is_dataframe:
        # Set title of figure
        fig.suptitle(df["product_hash"].iloc[0])

        plot_acf(df["sales_quantity"], lags=(len(df) - 1) - 1)
        plt.show()
        plot_pacf(df["sales_quantity"], lags=(len(df) - 1) / 2 - 1, method="ywm")
    else:
        # Set title of figure
        fig.suptitle(product_name)

        plot_acf(df, lags=(len(df) - 1) - 1)
        plt.show()
        plot_pacf(df, lags=(len(df) - 1) / 2 - 1, method="ywm")

    plt.show()

    plt.tight_layout()
    plt.show()
