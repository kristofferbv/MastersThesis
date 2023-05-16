import random

import retrieve_data
from analyse_data import plot_sales_quantity, get_non_stationary_products, decompose_sales_quantity
from config_utils import load_config
import os
import simulation
from generate_data import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':
    config = load_config("../config.yml")
    n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
    n_products = config["deterministic_model"]["n_products"]
    should_analyse = config["main"]["should_analyse"]
    use_stationary_data = config["main"]["stationary_products"]
    generate_new_data = config["main"]["generate_new_data"]


    # Reading the products created by the "read_products" function above
    if use_stationary_data:
        products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
        products = [df["sales_quantity"] for df in products]

    else:
        products = retrieve_data.read_products_3("2016-01-01", "2020-12-30")
        products = random.sample(get_non_stationary_products(products), n_products)
    if generate_new_data:
        products = generate_seasonal_data_based_on_products(products, 260)


    start_date = products[0].index[208]

    if should_analyse:  # analysing plotting, decomposing and testing for stationarity
        plot_sales_quantity(products)
        get_non_stationary_products(products, should_plot=True, verbose=True)
        for i, product in enumerate(products):
            if isinstance(product, pd.DataFrame):
                decompose_sales_quantity(product, product["product_hash"].iloc[0])
            else:
                decompose_sales_quantity(product, str(i))

    simulation.simulate(start_date, n_time_periods, products)

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
