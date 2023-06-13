import json
from datetime import timedelta, datetime
import random

import numpy as np
import pandas as pd
import os
import sys

from sklearn.preprocessing import OneHotEncoder

import deterministic_model as det_mod
import generate_data
from MIP.analysis.analyse_data import plot_sales_quantity
from MIP.forecasting import holt_winters_method
from MIP.standard_deviation import get_initial_std_dev, get_std_dev
from config_utils import load_config

# Get the path of the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the current path to the system path
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
config_path = os.path.join(parent_path, "config.yml")

config = load_config(config_path)
n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods we use in the deterministic model to decide actions
n_episodes = config["simulation"]["n_episodes"]  # This is the number of times we run a full simulation
simulation_length = config["simulation"]["simulation_length"]  # This is the number of time periods we want to calculate the costs for
start_index = 208

product_categories = config["deterministic_model"]["product_categories"]
seed = config["main"]["seed"]
n_products = sum(product_categories.values())

n_erratic = product_categories["erratic"]
n_smooth = product_categories["smooth"]
n_intermittent = product_categories["intermittent"]
n_lumpy = product_categories["lumpy"]

major_setup_cost = config["deterministic_model"]["joint_setup_cost"]
minor_setup_ratio = config["deterministic_model"]["minor_setup_ratio"]
beta = config["deterministic_model"]["beta"]


def simulate(real_products):
    output_folder = "results"
    current_datetime = datetime.now()
    current_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    output_file = f"train_data_p{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_t{n_time_periods}_ep{n_episodes}_S{major_setup_cost}_r{minor_setup_ratio}_time{current_datetime}.txt"
    file_path = os.path.join(output_folder, output_file)
    if os.path.exists(file_path):
        os.remove(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generated_products = []
    current_index = 0
    last_index = 0
    for category in product_categories.keys():
        number_of_products = product_categories[category]
        current_index += number_of_products
        if number_of_products > 0:
            generated_products += generate_data.generate_seasonal_data_for_erratic_demand(real_products[last_index:current_index], simulation_length * n_time_periods + start_index + 52)
            last_index = current_index
        # plot_sales_quantity(generated_products)
    start_date = generated_products[0].index[start_index]
    # plot_sales_quantity(generated_products)

    sample_data(file_path, start_date, n_time_periods, generated_products, simulation_length, real_products)


def sample_data(file_path, start_date, n_time_periods, products, episode_length, real_products):
    config = load_config("../config.yml")
    should_set_holding_cost_dynamically = config["simulation"]["should_set_holding_cost_dynamically"]
    if should_set_holding_cost_dynamically:
        unit_price = [df.iloc[0]['average_unit_price'] for df in products]


    inventory_levels = [0 for i in range(len(products))]

    dict_sds = {}
    dict_demands = {}
    prev_std_dev ={}
    prev_forecast ={}
    forecasts = {}


    for time_step in range(episode_length):
        print(f"Time step {time_step}/{episode_length}")
        actual_demands = []
        start_date += timedelta(weeks = 13)
        for product_index in range(len(products)):
            zero_data = pd.Series([0])
            sales_quantity_data = products[product_index].loc[start_date + pd.DateOffset(weeks=1):start_date + pd.DateOffset(weeks=n_time_periods + 1), "sales_quantity"]
            sales_quantity_data = pd.concat([zero_data, sales_quantity_data]).reset_index(drop=True)
            dict_demands[product_index] = sales_quantity_data

            if time_step == 0:
                dict_demands[product_index], train = holt_winters_method.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
                dict_sds[product_index] = get_initial_std_dev(train, n_time_periods)
                # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                prev_std_dev[product_index] = dict_sds[product_index][1]
                prev_forecast[product_index] = dict_demands[product_index][1]
            else:
                demand = products[product_index].loc[date_time, "sales_quantity"]
                forecast_errors = abs(demand - prev_forecast[product_index])
                dict_sds[product_index] = get_std_dev(prev_std_dev[product_index], forecast_errors, n_time_periods, alpha=0.1)
                dict_demands[product_index], _ = holt_winters_method.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
                # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                prev_std_dev[product_index] = dict_sds[product_index][1]
                prev_forecast[product_index] = dict_demands[product_index][1]
        major_costs = random.randint(100,2500)
        print(major_costs)
        deterministic_model = det_mod.DeterministicModel(len(products))
        deterministic_model.set_costs(major_costs)
        deterministic_model.set_demand_forecast(dict_demands)
        if should_set_holding_cost_dynamically:
            deterministic_model.set_holding_costs(unit_price)
        deterministic_model.set_big_m()
        deterministic_model.model.setParam("OutputFlag", 0)
        # deterministic_model.model.setParam('TimeLimit', 2 * 60)  # set the time limit to 2 minutes for the gurobi model
        # deterministic_model.model.setParam('MIPGap', 0.01)  # set the MIPGap to be 1%

        deterministic_model.set_inventory_levels(inventory_levels)
        deterministic_model.set_up_model()
        deterministic_model.optimize()

        # Extract and store the first action for each product in the current time step
        threshold = 1e-5
        # Initialize empty lists
        # Initialize dictionaries to store states and targets for each product
        states_dict = {}
        targets_dict = {}
        # Loop over products and time steps
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        for product_index in range(len(products)):
            product_states = []
            product_targets = []
            for time in range(1, 13+1):
                # Forecast
                order_product_value = [0] * 13  # Initialize a list of 13 zeros
                hei = 1
                has_set = False
                for tau_period in range(1,13 - time + 2):
                    # OrderProduct
                    if deterministic_model.model.getVarByName(f"OrderProduct[{product_index},{time},{tau_period}]").X > threshold:
                        # order_product_value[tau_period] = 1
                        hei = tau_period
                        has_set = True
                if not has_set:
                    order_product_value[0] = 1
                    hei = 1

                forecasted_date = sum(dict_demands[product_index][time: time + hei])
                # DateTime
                date_time = start_date + pd.DateOffset(weeks=time)
                # Month
                month = date_time.month
                # one_hot_month = one_hot_encoder.fit_transform(np.array(month).reshape(-1, 1)).tolist()[0]

                # Append the state
                product_states.append([forecasted_date, time, hei, month])
                # # Append the state
                # product_states.append([forecasted_date, time_step, hei])

                # Actual demand
                demand = sum(list(products[product_index].loc[date_time: date_time + pd.DateOffset(weeks=hei), "sales_quantity"]))

                # Target (actual demand - forecast)
                target = abs(demand - forecasted_date)

                # Append the target
                product_targets.append(target)

            # Store the states and targets for each product in the dictionaries
            states_dict[product_index] = product_states
            targets_dict[product_index] = product_targets

        # Loop over each product index and write the states and targets to separate files
        for product_index in range(len(products)):
            with open(f"results/states{product_index}.txt", "a") as f:  # "a" for appending to the file
                f.write(f"{json.dumps(states_dict[product_index])}\n")  # Added newline character
            with open(f"results/targets{product_index}.txt", "a") as f:  # "a" for appending to the file
                f.write(f"{json.dumps(targets_dict[product_index])}\n")  # Added newline character
