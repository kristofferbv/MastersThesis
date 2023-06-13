import random
import os
import sys

from MIP.analysis.analyse_data import plot_sales_quantity
from MIP.forecasting import sarima, holt_winters_method, crostons_method, recurrent_neural_network, benchmark_avg

import retrieve_data
from config_utils import load_config
from generate_data import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':
    product_categories = {"erratic": 10, "smooth": 0, "intermittent": 0, "lumpy": 0}
    seed = 3

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
        for i in range(10):
            category_products = retrieve_data.read_products("2016-01-01", "2020-12-30", category)
            number_of_products = 20
            sampled_products = random.sample(category_products, number_of_products)
            if category == "erratic" or category == "smooth":
                products = generate_seasonal_data_for_erratic_demand(sampled_products, 10000)
            else:
                products = generate_seasonal_data_for_intermittent_demand(sampled_products, 10000)
            # products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
            # products = generate_seasonal_data_based_on_products(sampled_products, 350, period=12)
            # plot_sales_quantity(products)            #
            # start_date = products[0].index[48]
            # print(products[0])
            start_date = products[0].index[8000]
            # start_date = products[0].index[random.randint(208, 260)]


            for product in products:
                h_mae, h_mse, h_rmse = holt_winters_method.forecast_analysis(product, start_date, verbose=False, n_time_periods=26, shouldShowPlot=True)
                if (h_mae < 100):
                    h_mses.append(h_mse)
                    h_maes.append(h_mae)
                    h_rmses.append(h_rmse)
                    print(f"Average MSE holt_winter = {sum(h_mses) / len(h_mses)}")
                    print(f"Std dev MSE holt_winter = {np.std(h_mses)}")
                    print(f"Average MAE holt_winter = {sum(h_maes) / len(h_maes)}")
                    # c_mae, c_mse, c_rmse = crostons_method.forecast_analysis(product, start_date, verbose=False, n_time_periods=26, shouldShowPlot=False)
                    # c_mses.append(c_mse)
                    # c_maes.append(c_mae)
                    # c_rmses.append(c_rmse)

                    r = recurrent_neural_network.forecast(product,start_date, n_time_periods=26)
                    # s_mae, s_mse, s_rmse = sarima.forecast_analyse(product, start_date, n_time_periods=26)
                    # s_mses.append(s_mse)
                    # s_maes.append(s_mae)
                    # s_rmses.append(s_rmse)
                    # b_mae, b_mse, b_rmse = benchmark_avg.forecast_analysis(product, start_date, n_time_periods=26)
                    # b_mses.append(b_mse)
                    # b_maes.append(b_mae)
                    # b_rmses.append(b_rmse)

        # print()
        # print("***************************************")
        # print(f"statistics for category: {category}")
        # print()
        # print(f"Average MSE holt_winter = {sum(h_mses) / len(h_mses)}")
        # print(f"Std dev MSE holt_winter = {np.std(h_mses)}")
        # print(f"Average MAE holt_winter = {sum(h_maes) / len(h_maes)}")
        # print(f"Std dev MAE holt_winter = {np.std(h_maes)}")
        # print(f"Average RMSE holt_winter = {sum(h_rmses) / len(h_rmses)}")
        # print(f"Std dev RMSE holt_winter = {np.std(h_rmses)}")
        # print()
        # print(f"Average MSE Croston = {sum(c_mses) / len(c_mses)}")
        # print(f"Std dev MSE Croston = {np.std(c_mses)}")
        # print(f"Average MAE Croston = {sum(c_maes) / len(c_maes)}")
        # print(f"Std dev MAE Croston = {np.std(c_maes)}")
        # print(f"Average RMSE Croston = {sum(c_rmses) / len(c_rmses)}")
        # print(f"Std dev RMSE Croston = {np.std(c_rmses)}")
        # print()
        # print(f"Average MSE SARIMA = {sum(s_mses) / len(s_mses)}")
        # print(f"Std dev MSE SARIMA = {np.std(s_mses)}")
        # print(f"Average MAE SARIMA = {sum(s_maes) / len(s_maes)}")
        # print(f"Std dev MAE SARIMA = {np.std(s_maes)}")
        # print(f"Average RMSE SARIMA = {sum(s_rmses) / len(s_rmses)}")
        # print(f"Std dev RMSE SARIMA = {np.std(s_rmses)}")
        # print()
        # print(f"Average MSE Benchmark = {sum(b_mses) / len(b_mses)}")
        # print(f"Std dev MSE Benchmark = {np.std(b_mses)}")
        # print(f"Average MAE Benchmark = {sum(b_maes) / len(b_maes)}")
        # print(f"Std dev MAE Benchmark = {np.std(b_maes)}")
        # print(f"Average RMSE Benchmark = {sum(b_rmses) / len(b_rmses)}")
        # print(f"Std dev RMSE Benchmark = {np.std(b_rmses)}")
        # print("***************************************")
        # print()


