import retrieve_data
from analyse_data import plot_sales_quantity, get_non_stationary_products, decompose_sales_quantity
import holt_winters_method, sarima, recurrent_neural_network
from config_utils import load_config
import os
import simulation

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':
    config = load_config("config.yml")
    n_time_periods = config["n_time_periods"]  # number of time periods

    # retrieve_data.categorize_products("data/sales_orders.csv", "m", True)
    products = retrieve_data.read_products_3("2016-01-01", "2020-12-30")
    print("number of products: ", len(products))

    # Reading the products created by the "read_products" function above
    products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
    start_date = products[0].index[108]

    should_analyse = False
    if should_analyse:  # analyse data and retrieve non_stationary products
        plot_sales_quantity(products[:20])
        get_non_stationary_products(products)
        for product in products:
            decompose_sales_quantity(product)

    # List of non-stationary products based on the analysis above
    # non_stationary_products = ["0c0f3efa3afddcae74bf01414219044b", "0cf88020722953499b7e6ee70c16f36b", "3c50c0cf057cb8aab8bf3fb28b711b6a", "4151b0029636a1c55afcce9283ac7902" , "6dcb66034aed7493a93ef9b231ecaf14", "af5ed76b466a037cd7b9b1cefef578ba" ,"bfaac30872cb86835b1fd11b4e4129d8" , "edb636f69bf78b885117a47ec1a455d4"]

    # sarima.forecast(non_stationary_products[0], start_date)
    # holt_winters_method.forecast(products[0]["sales_quantity"], start_date, shouldShowPlot=True)
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
