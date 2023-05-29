import random

import os
import sys

# Get the path of the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the current path to the system path
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)


import retrieve_data
from analyse_data import plot_sales_quantity, get_non_stationary_products, decompose_sales_quantity
from config_utils import load_config
import simulation
from generate_data import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)



if __name__ == '__main__':
    config = load_config("config.yml")
    n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
    n_products = config["deterministic_model"]["n_products"]
    should_analyse = config["main"]["should_analyse"]
    use_stationary_data = config["main"]["stationary_products"]
    generate_new_data = config["main"]["generate_new_data"]


    # Reading the products created by the "read_products" function above
    if use_stationary_data:
        products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
        #products = [df["sales_quantity"] for df in products]

        # comment line above and uncomment these to get random products instead of the 6 that are specificed
        #products = retrieve_data.read_products_3("2016-01-01", "2020-12-30")
        #products = random.sample(products, n_products) 


    else:
        products = retrieve_data.read_products_3("2016-01-01", "2020-12-30")
        products = random.sample(get_non_stationary_products(products), n_products)



    start_date = products[0].index[208]

    if should_analyse:  # analysing plotting, decomposing and testing for stationarity
        plot_sales_quantity(products)
        get_non_stationary_products(products, should_plot=True, verbose=True)
        for i, product in enumerate(products):
            if isinstance(product, pd.DataFrame):
                decompose_sales_quantity(product, product["product_hash"].iloc[0])
            else:
                decompose_sales_quantity(product, str(i))
    if generate_new_data:
        simulation.simulate(products)
    else:
        simulation_length = config["simulation"]["simulation_length"]
        simulation.run_one_episode(start_date, n_time_periods, simulation_length, products)