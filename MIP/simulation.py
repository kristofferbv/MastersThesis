from datetime import timedelta

import numpy as np
import pandas as pd
import os
import sys
import time

import deterministic_model as det_mod
import sarima
import holt_winters_method
from config_utils import load_config
import generate_data
from generate_data import generate_seasonal_data_based_on_products

# Get the path of the current script
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory of the current path to the system path
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
config_path = os.path.join(parent_path, "config.yml")


config = load_config(config_path)
n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods we use in the deterministic model to decide actions
n_episodes = config["simulation"]["n_episodes"] # This is the number of times we run a full simulation
simulation_length = config["simulation"]["simulation_length"] # This is the number of time periods we want to calculate the costs for
warm_up_length = config["simulation"]["warm_up_length"] # This is the number of time periods we are using to warm up
should_perform_warm_up = config["simulation"]["should_perform_warm_up"]
reset_length =  config["simulation"]["reset_length"]
start_index = 105

def simulate(real_products):
    total_costs = []
    inventory_levels = None
    print("GEE", (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods)
    generated_products = generate_seasonal_data_based_on_products(real_products, (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods)
    print(len(generated_products[0]))
    start_date = generated_products[0].index[start_index]
    if should_perform_warm_up:
        print("Warming up")
        inventory_levels, start_date = perform_warm_up(generated_products, start_date, n_time_periods)
    for episode in range(n_episodes):
            # simulate and sample costs
            print("Running simulation...")
            costs, inventory_levels, _ , _ = run_one_episode(start_date, n_time_periods, generated_products, simulation_length, inventory_levels=inventory_levels)
            total_costs.append(costs)
            print(f"Costs for episode {episode} is: {costs}")
            print("Resetting...")
            # resetting
            costs, inventory_levels, _ , _ = run_one_episode(start_date, n_time_periods, generated_products, reset_length, inventory_levels=inventory_levels)
    print(f"Total average costs for all episodes is: {sum(total_costs)/len(total_costs)}")

def perform_warm_up(products, start_date, n_time_periods):
    inventory_levels = [0 for i in range(len(products))]
    #for i in range(simulation_length):
    _, inventory_levels, start_date, _ = run_one_episode(start_date, n_time_periods, products, warm_up_length, inventory_levels=inventory_levels)
    return inventory_levels, start_date

def run_one_episode(start_date, n_time_periods, products, episode_length,  inventory_levels = None):

    start_time = time.time()

    config = load_config("../config.yml")
    forecasting_method = config["simulation"]["forecasting_method"]  # number of time periods
    verbose = config["simulation"]["verbose"]  # number of time periods
    should_set_holding_cost_dynamically = config["simulation"]["should_set_holding_cost_dynamically"]
    if should_set_holding_cost_dynamically:
        unit_costs = [df.iloc[0]['average_unit_cost'] for df in products]

    dict_demands = {}
    dict_sds = {}
    actions = {}  # Store the first actions for each time step
    orders = {}
    if inventory_levels is None:
        inventory_levels = [0 for i in range(len(products))]

    total_costs = 0

    shortage_costs = 0
    holding_costs = 0
    setup_costs = 0

    sum_actual_demand = {product_index: 0 for product_index in range(len(products))}
    sum_fulfilled_demand = {product_index: 0 for product_index in range(len(products))}


    for time_step in range(episode_length):
        print(f"Time step {time_step}/{episode_length}")
        start_date = start_date + timedelta(days=7)

        period_costs = 0
        major_setup_added = False

        # Update inventory levels based on previous actions and actual demand
        actual_demands = []
        if time_step != 0:
            for product_index, product in enumerate(products):
                if isinstance(product, pd.DataFrame):
                    demand = products[product_index].loc[start_date, "sales_quantity"]
                else:
                    demand = products[product_index].loc[start_date]

                actual_demands.append(demand)

                # used to calculate the service level
                sum_actual_demand[product_index] += demand
                sum_fulfilled_demand[product_index] += min(inventory_levels[product_index] + actions[time_step - 1][product_index], demand)

                # add holding costs or shortage costs
                if inventory_levels[product_index] + actions[time_step - 1][product_index] - demand > 0:
                    holding_costs += (inventory_levels[product_index] + actions[time_step - 1][product_index] - demand) * deterministic_model.holding_cost[product_index]
                    period_costs += (inventory_levels[product_index] + actions[time_step - 1][product_index] - demand) * deterministic_model.holding_cost[product_index]
                else:
                    shortage_costs += abs(inventory_levels[product_index] + actions[time_step - 1][product_index] - demand) * deterministic_model.shortage_cost[product_index]
                    period_costs += abs(inventory_levels[product_index] + actions[time_step - 1][product_index] - demand) * deterministic_model.shortage_cost[product_index]

                # add setup costs:
                if actions[time_step - 1][product_index] > 0:
                    setup_costs += deterministic_model.minor_setup_cost[product_index]
                    period_costs += deterministic_model.minor_setup_cost[product_index]

                    # to only add major setup costs once if an order is made
                    if not major_setup_added:
                        setup_costs += deterministic_model.major_setup_cost
                        period_costs += deterministic_model.major_setup_cost
                        major_setup_added = True

                previous_il = inventory_levels[product_index]
                inventory_levels[product_index] = max(0, previous_il + actions[time_step - 1][product_index] - demand)

            total_costs += period_costs
            if verbose:
                print("Period costs: ")
                print(period_costs)

                print("Actions at time period ", time_step - 1)
                print(actions[time_step - 1])

                print("Actual_demand for period ", time_step - 1)
                print(actual_demands)

                print("Inventory levels at start of time period ", time_step)
                print(inventory_levels)

                print("Total costs at time period : ", time_step)
                print(total_costs)

                print("Total holding costs:")
                print(holding_costs)

                print("Total shortage costs:")
                print(shortage_costs)

                print("Total setup costs:")
                print(setup_costs)

        for product_index in range(len(products)):
            if forecasting_method == "holt_winter":
                dict_demands[product_index], dict_sds[product_index] = holt_winters_method.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
            elif forecasting_method == "sarima":
                dict_demands[product_index], dict_sds[product_index] = sarima.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
            else:
                raise ValueError(f"Forecasting method must be either 'sarima' or 'holt_winter', but is: {forecasting_method}")

        deterministic_model = det_mod.DeterministicModel(len(products))
        deterministic_model.set_demand_forecast(dict_demands)
        if should_set_holding_cost_dynamically:
            deterministic_model.set_holding_costs(unit_costs)
        deterministic_model.set_safety_stock(dict_sds)
        deterministic_model.set_big_m()
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.set_inventory_levels(inventory_levels)
        deterministic_model.set_up_model()
        deterministic_model.optimize()

        # Extract and store the first action for each product in the current time step
        actions[time_step] = {}
        threshold = 1e-5

        orders[time_step] = {}

        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                # Only looking at the action at time t = 1, since that is the actual action for this period
                if current_time == 1:
                    actions[time_step][product_index] = var.x
                    if abs(actions[time_step][product_index]) < threshold:
                        actions[time_step][product_index] = 0

            if var.varName.startswith("OrderProduct"):
                for tau in deterministic_model.tau_periods:
                    product_index, current_time, tau = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                    # Only looking at the action at time t = 1, since that is the actual action for this period
                    if current_time == 1:
                        if product_index not in orders[time_step]:
                            orders[time_step][product_index] = {}
                        orders[time_step][product_index][tau] = var.x
                        if abs(orders[time_step][product_index][tau]) < threshold:
                            orders[time_step][product_index][tau] = 0

   
    # print("Total costs at after all periods : ")
    # print(total_costs)
    # print("Total shortage costs")
    # print(shortage_costs)
    # print("Holding costs:")
    # print(holding_costs)
    # print("Setup costs")
    # print(setup_costs)
    # print(actions)
    # print("orders")
    # print(orders)
    # runtime = deterministic_model.model.Runtime
    # print("The run time is %f" % runtime)

    end_time = time.time()  # Stop measuring the time
    runtime = end_time - start_time
    print(f"Solution time for this episode is: {runtime} seconds")

    for product_index in range(len(products)):
        service_level = sum_fulfilled_demand[product_index] / sum_actual_demand[product_index]
        print(f"Achieved service level for Product {product_index}: {service_level}")

    
    print(actions)

    return total_costs, inventory_levels, start_date, actions



