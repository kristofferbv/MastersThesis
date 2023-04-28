import retrieve_data
import holt_winters_method, arima, recurrent_neural_network
from config_utils import load_config
import simulation
import deterministic_model
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)


if __name__ == '__main__':
    # deterministic_model = MIP.deterministic_model.DeterministicModel()
    # deterministic_model.set_up_model()
    # deterministic_model.model.optimize()

    config = load_config("config.yml")
    n_time_periods = config["n_time_periods"]  # number of time periods
    products = retrieve_data.read_products("2016-01-01", "2020-12-30")
    products2 = retrieve_data.read_products_2("2016-01-01", "2020-12-30")

    start_date = products[0].index[208]
    print(start_date)
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
