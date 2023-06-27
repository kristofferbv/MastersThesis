import random
import os
import sys
import warnings

from MIP import simulate_states

warnings.filterwarnings("ignore")
# Get the path of the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the current path to the system path
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import retrieve_data
from MIP.analysis.analyse_data import plot_sales_quantity, get_non_stationary_products, decompose_sales_quantity
from config_utils import load_config
import simulation
from generate_data import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

if __name__ == '__main__':

    # config_files = ["config.yml", "config2.yml", "config3.yml", "config4.yml", "config5.yml", "config6.yml", "config7.yml","config8.yml","config9.yml","config10.yml"]

    config_files = ["config.yml"]

    for config_file in config_files:
        read_product_hashes = {}
        read_product_hashes["erratic"] = ["1fb096daa569c811723ce8796722680e", "ede08f3ae4d17d223051ed8282a67caf", "14c82a06c71b2f73856ffccc166c9e35", "bdc9f6ada518791b8b76e8a07021e145"]
        read_product_hashes["smooth"] = ["3b777b775721dfa8d36de2a320a03e53", "790d67b8e374ac145b107e84b846ebd5", "5c4f77b0c88aa306e667e6096430b339", "c7d67339105519c3dae1c323022dbd04"]
        read_product_hashes["intermittent"] = ["99de18277c25fa7fe70c02a0039a2141", "460066ec29274e9c86d952a27dc83f5f", "73bcaace6e64df5e9c6c3997199c491f", "fa5723a60fc448c4ee84819937dfdeb7"]
        read_product_hashes["lumpy"] = ["9a253ffc9db1a7849b1bb8adccb36709", "52f944862e84ac0972f866a99083e9ce", "4e631fda7a54afb051c1fcb57d9e592a", "08797c9a781ba4e5246b9bed47ebf0fd"]
        config = load_config(config_file)
        n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
        should_analyse = config["main"]["should_analyse"]
        use_stationary_data = False  # config["main"]["stationary_products"]
        generate_new_data = config["main"]["generate_new_data"]
        product_categories = config["deterministic_model"]["product_categories"]
        seed = config["main"]["seed"]
        n_products = sum(product_categories.values())
        # retrieve_data.categorize_products("sales_orders.csv", "w", True)

        # calculate average unit costs to compute setup costs
        all_products = retrieve_data.read_products("2017-01-01", "2020-12-30")

        unit_price_all = [df.iloc[0]['average_unit_price'] for df in all_products]

        average_unit_price = sum(unit_price_all) / len(unit_price_all)

        print("The average unit cost is: ", average_unit_price)

        # Reading the products created by the "read_products" function above
        products = []
        if seed is not None:
            # Setting a random seed ensure we select the same random products each time
            random.seed(seed)

        if False:
            for category in read_product_hashes:
                category_products = retrieve_data.read_products_with_hashes_2("2017-01-01", "2020-12-30", read_product_hashes[category], category)
                products += category_products
                print("len", len(products))

        else:
            for category in product_categories.keys():
                category_products = retrieve_data.read_products("2017-01-01", "2020-12-30", category)
                category_products = [product for product in category_products if product["sales_quantity"].max() <= 150]
                category_products.sort(key=lambda product: product["sales_quantity"].mean())

                number_of_products = product_categories[category]

                if number_of_products > 0 and len(category_products) > 0:
                    # Make sure the number of products required does not exceed the number of available products
                    number_of_products = min(number_of_products, len(category_products))

                    products += random.sample(category_products, number_of_products)
                    # products += category_products[-2:]
                print("len", len(products))

        # products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
        # products = [df["sales_quantity"] for df in products]
        plot_sales_quantity(products)

        # products = generate_seasonal_data_based_on_products(products, 500, seed = 1)
        # sarima.forecast(products[0], start_date, 20)
        # holt_winters_method.forecast(products[0], start_date, True, 20)

        if should_analyse:  # analysing plotting, decomposing and testing for stationarity
            plot_sales_quantity(products)
            get_non_stationary_products(products, should_plot=True, verbose=True)
            for i, product in enumerate(products):
                if isinstance(product, pd.DataFrame):
                    decompose_sales_quantity(product, product["product_hash"].iloc[0])
                else:
                    decompose_sales_quantity(product, str(i))

        if generate_new_data:
            #print("Products")
            #print(len(products))
            simulation.simulate(products, config)
            # simulate_states.simulate(products)

        else:
            start_date = products[0].index[104]
            simulation_length = config["simulation"]["simulation_length"]
            simulation.run_one_episode(start_date, n_time_periods, simulation_length, products, config)
