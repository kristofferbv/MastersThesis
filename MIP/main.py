import random

import retrieve_data
from analyse_data import plot_sales_quantity, get_non_stationary_products, decompose_sales_quantity
from config_utils import load_config
import os
import simulation
from generate_data import *
import holt_winters_method
import sarima

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':
    config = load_config("../config.yml")
    n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
    should_analyse = config["main"]["should_analyse"]
    use_stationary_data = False # config["main"]["stationary_products"]
    generate_new_data = config["main"]["generate_new_data"]
    product_categories = config["main"]["product_categories"]
    seed = config["main"]["seed"]
    n_products = sum(product_categories.values())


    # Reading the products created by the "read_products" function above
    products = []
    if seed is not None:
        np.random.seed(seed)
    for category in product_categories.keys():
        products.extend(retrieve_data.read_products("2016-01-01", "2020-12-30", category))
        print(category)
        number_of_products = product_categories[category]
        print(number_of_products)
        if number_of_products>0:
            products = random.sample(products, product_categories[category])

    # products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
    products = [df["sales_quantity"] for df in products]

    print(len(products[0]))
    start_date = products[0].index[208]
    # products = generate_seasonal_data_based_on_products(products, 500, seed = 1)
    sarima.forecast(products[0], start_date, 20)
    holt_winters_method.forecast(products[0], start_date, True, 20)



    #
    # if should_analyse:  # analysing plotting, decomposing and testing for stationarity
    #     plot_sales_quantity(products)
    #     get_non_stationary_products(products, should_plot=True, verbose=True)
    #     for i, product in enumerate(products):
    #         if isinstance(product, pd.DataFrame):
    #             decompose_sales_quantity(product, product["product_hash"].iloc[0])
    #         else:
    #             decompose_sales_quantity(product, str(i))
    # if generate_new_data:
    #     simulation.simulate(products[:4])
    # else:
    #     simulation.run_one_episode(start_date, n_time_periods, products)