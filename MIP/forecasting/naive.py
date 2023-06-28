import statistics
import numpy as np
import matplotlib.pyplot as plt
def forecast_analysis(df, date, shouldShowPlot=False, verbose = False, n_time_periods=20):
    # Shift the 'sales_quantity' by 52 weeks
    shifted_df = df["sales_quantity"].shift(52)


    # Split the data into training and testing sets
    train = df.loc[df.index <= date]
    test = df.loc[df.index > date]

    # For the forecast, take the 'sales_quantity' from the same period last year
    forecast = shifted_df.loc[test.index]

    # Ensure forecast is non-negative
    forecast[forecast < 0] = 0

    if shouldShowPlot:
        # Evaluate the forecast
        plt.plot(test.index[:n_time_periods], test["sales_quantity"].values[:n_time_periods], label='Actual')
        plt.plot(test.index[:n_time_periods], forecast.values[:n_time_periods], label='Forecast')

        # Add labels and legend
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title('Sales Forecast using Naïve method')
        plt.legend()

        # Show the plot
        plt.show()

    mse = np.mean((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods]) ** 2)
    mae = np.mean(np.abs((test["sales_quantity"].values[:n_time_periods] - forecast[:n_time_periods])))
    rmse = np.sqrt(mse)

    if verbose:
        print(f'MAE for Naïve method: {mae:.2f}')

    return mae, mse, rmse, forecast.values[:n_time_periods], test["sales_quantity"].values[:n_time_periods], test.index[:n_time_periods]
