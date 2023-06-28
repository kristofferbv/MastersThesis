import random
import os
import sys

from MIP.forecasting import sarima, holt_winters_method, crostons_method, recurrent_neural_network, benchmark_avg, naive, TSB

import retrieve_data
from config_utils import load_config
from generate_data import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

import matplotlib.font_manager

plt.rcParams["font.family"] = "CMU Concrete"
plt.rcParams["font.family"] = "CMU Concrete"

mpl.rc('font', family='CMU Concrete')
plt.rcParams['font.size'] = 11


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':
    product_categories = {"lumpy":0}
    seed = None

    # Reading the products created by the "read_products" function above
    products = []
    if seed is not None:
        # Setting a random seed ensure we select the same random products each time
        random.seed(seed)
    for category in product_categories.keys():
        s_mses = []
        s_maes = []
        s_rmses = []

        h_mses = []
        h_maes = []
        h_rmses = []

        c_maes = []
        c_mses = []
        c_rmses = []

        b_maes = []
        b_mses = []
        b_rmses = []

        r = []
        # for i in range(10):
        category_products = retrieve_data.read_products("2016-01-01", "2020-12-30", category)
        number_of_products = min(len(category_products), 15)
        sampled_products = random.sample(category_products, number_of_products)
        if category == "erratic" or category == "smooth":
            products = generate_seasonal_data_based_on_products(sampled_products, 350)
        else:
            products = generate_seasonal_data_for_intermittent_demand(sampled_products, 350)

        start_date = products[0].index[random.randint(208, 260)]

        for product in products:
            h_mae, h_mse, h_rmse,f1 = holt_winters_method.forecast_analysis(product, start_date, verbose=False, n_time_periods=26)
            if (h_mae < 100):
                h_mses.append(h_mse)
                h_maes.append(h_mae)
                h_rmses.append(h_rmse)
                c_mae, c_mse, c_rmse, f2 = TSB.forecast_analysis(product, start_date, verbose=False, n_time_periods=26)
                c_mses.append(c_mse)
                c_maes.append(c_mae)
                c_rmses.append(c_rmse)
                # r = recurrent_neural_network.forecast(product,start_date, n_time_periods=26)
                s_mae, s_mse, s_rmse, f3 = sarima.forecast_analyse(product, start_date, n_time_periods=26)
                s_mses.append(s_mse)
                s_maes.append(s_mae)
                s_rmses.append(s_rmse)
                b_mae, b_mse, b_rmse,f4 ,t, x = naive.forecast_analysis(product, start_date, n_time_periods=26)
                b_mses.append(b_mse)
                b_maes.append(b_mae)
                b_rmses.append(b_rmse)

                plt.plot(x,t, label='Actual Demand')

                # Plot forecasts
                plt.plot(x, f1, label='Holt-Winter')
                plt.plot(x, f3, label='SARIMA')
                plt.plot(x, f4, label='Naive')
                plt.plot(x, f2, label='TSB')

                # Add title and labels
                plt.xlabel('Date')
                plt.ylabel('Demand')

                # Add legend to distinguish different lines
                plt.legend()

                # Display the plot
                plt.show()


        print()
        print("***************************************")
        print(f"statistics for category: {category}")
        print()
        print(f"Average MSE holt_winter = {sum(h_mses) / len(h_mses)}")
        print(f"Std dev MSE holt_winter = {np.std(h_mses)}")
        print(f"Average MAE holt_winter = {sum(h_maes) / len(h_maes)}")
        print(f"Std dev MAE holt_winter = {np.std(h_maes)}")
        print(f"Average RMSE holt_winter = {sum(h_rmses) / len(h_rmses)}")
        print(f"Std dev RMSE holt_winter = {np.std(h_rmses)}")
        print()
        print(f"Average MSE  TSB = {sum(c_mses) / len(c_mses)}")
        print(f"Std dev MSE  TSB = {np.std(c_mses)}")
        print(f"Average MAE  TSB = {sum(c_maes) / len(c_maes)}")
        print(f"Std dev MAE  TSB = {np.std(c_maes)}")
        print(f"Average RMSE TSB = {sum(c_rmses) / len(c_rmses)}")
        print(f"Std dev RMSE TSB = {np.std(c_rmses)}")
        print()
        print(f"Average MSE SARIMA = {sum(s_mses) / len(s_mses)}")
        print(f"Std dev MSE SARIMA = {np.std(s_mses)}")
        print(f"Average MAE SARIMA = {sum(s_maes) / len(s_maes)}")
        print(f"Std dev MAE SARIMA = {np.std(s_maes)}")
        print(f"Average RMSE SARIMA = {sum(s_rmses) / len(s_rmses)}")
        print(f"Std dev RMSE SARIMA = {np.std(s_rmses)}")
        print()
        print(f"Average MSE Benchmark  Naïve = {sum(b_mses) / len(b_mses)}")
        print(f"Std dev MSE Benchmark  Naïve = {np.std(b_mses)}")
        print(f"Average MAE Benchmark  Naïve = {sum(b_maes) / len(b_maes)}")
        print(f"Std dev MAE Benchmark  Naïve = {np.std(b_maes)}")
        print(f"Average RMSE Benchmark Naïve = {sum(b_rmses) / len(b_rmses)}")
        print(f"Std dev RMSE Benchmark Naïve = {np.std(b_rmses)}")
        print("***************************************")
        print()
