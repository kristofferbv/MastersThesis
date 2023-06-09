from datetime import timedelta

import pandas as pd
import os
import sys
import time
import statistics

import deterministic_model as det_mod
from MIP.forecasting import holt_winters_method, sarima
from MIP.standard_deviation import get_initial_std_dev, get_std_dev
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
n_episodes = config["simulation"]["n_episodes"]  # This is the number of times we run a full simulation
simulation_length = config["simulation"]["simulation_length"]  # This is the number of time periods we want to calculate the costs for
warm_up_length = config["simulation"]["warm_up_length"]  # This is the number of time periods we are using to warm up
should_perform_warm_up = config["simulation"]["should_perform_warm_up"]
reset_length = config["simulation"]["reset_length"]
should_write = config["simulation"]["should_write"]
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

    output_file = f"simulation_output_p{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_t{n_time_periods}_ep{n_episodes}_S{major_setup_cost}_r{minor_setup_ratio}_beta{beta}_seed{seed}.txt"
    file_path = os.path.join(output_folder, output_file)
    if os.path.exists(file_path):
        os.remove(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    generated_products = []
    current_index = 0
    last_index = 0
    with open(file_path, "a") as f:
        for category in product_categories.keys():
            number_of_products = product_categories[category]
            current_index += number_of_products
            f.write("Number of products of category " + category + " is " + str(number_of_products) + "\n")
            # if category == "erratic" or category == "smooth":
            #     generated_products += generate_seasonal_data_based_on_products(real_products[last_index:current_index], (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods + 52)
            # else:
            #     print("FUCK!!")
            #     print(len(generated_products))
            #     generated_products += generate_data.generate_seasonal_data_for_intermittent_demand(real_products[last_index:current_index], (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods + 52)
            #     print(len(generated_products))
            generated_products += generate_seasonal_data_based_on_products(real_products[last_index:current_index], (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods + 52)
            last_index = current_index

        total_costs = []
        list_mean = []
        list_std = []
        inventory_levels = None
        start_date = generated_products[0].index[start_index]

        models = {}
        if should_perform_warm_up and warm_up_length > 0:
            print("Warming up")
            inventory_levels, start_date, models = perform_warm_up(generated_products, start_date, n_time_periods)
        for episode in range(n_episodes):
            # simulate and sample costs
            print("Running simulation...")
            costs, inventory_levels, start_date, _, models = run_one_episode(start_date, n_time_periods, generated_products, simulation_length, models=models, inventory_levels=inventory_levels)
            total_costs.append(costs)
            list_mean.append(sum(total_costs) / len(total_costs))
            if episode > 0:
                list_std.append(statistics.stdev(total_costs))
            print(f"Costs for episode {episode} is: {costs}")
            # f.write(f"Actions for episode {episode} are: {actions}" + "\n")
            # print(f"Actions for episode {episode} are: {actions}")
            if reset_length > 0:
                print("Resetting...")
                # resetting
                costs, inventory_levels, start_date, _, models = run_one_episode(start_date, n_time_periods, generated_products, reset_length, models=models, inventory_levels=inventory_levels)
        if should_write:
            f.write(f"Total average costs for all episodes is: {sum(total_costs) / len(total_costs)}" + "\n")
            f.write(f'List of mean for each period: {list_mean}' + "\n")
            f.write(f'List of standard deviation for each period: {list_std}' + "\n")

            standard_deviation_costs = statistics.stdev(total_costs)
            f.write(f"Standard deviations of costs: {standard_deviation_costs}" + "\n")
            f.close()
        print(f"Total average costs for all episodes is: {sum(total_costs) / len(total_costs)}")


def perform_warm_up(products, start_date, n_time_periods):
    inventory_levels = [0 for i in range(len(products))]
    # for i in range(simulation_length):
    _, inventory_levels, start_date, _, models = run_one_episode(start_date, n_time_periods, products, warm_up_length, inventory_levels=inventory_levels)
    return inventory_levels, start_date, models


def run_one_episode(start_date, n_time_periods, products, episode_length, models=None, inventory_levels=None):
    start_time = time.time()

    config = load_config("../config.yml")
    forecasting_method = config["simulation"]["forecasting_method"]  # number of time periods
    verbose = config["simulation"]["verbose"]  # number of time periods
    should_set_holding_cost_dynamically = config["simulation"]["should_set_holding_cost_dynamically"]
    if should_set_holding_cost_dynamically:
        unit_price = [df.iloc[0]['average_unit_price'] for df in products]
    if models is None:
        models = {}
    dict_demands = {}
    dict_sds = {}
    actions = {}  # Store the first actions for each time step
    orders = {}
    forecast_errors = {}
    prev_std_dev = {}
    prev_forecast = {}
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
                print("DEMAAAND", demand)
                print(start_date)

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
        for product_index in range(len(products)):
            if forecasting_method == "holt_winter":
                sales_quantity_data = products[product_index].loc[start_date + pd.DateOffset(weeks=1):start_date + pd.DateOffset(weeks=13), "sales_quantity"]

                print("sales:", sales_quantity_data)
                print(holt_winters_method.forecast(products[product_index],start_date, 13))
                dict_demands[product_index] = sales_quantity_data
                print(start_date)

        deterministic_model = det_mod.DeterministicModel(len(products))
        deterministic_model.set_demand_forecast(dict_demands)
        if should_set_holding_cost_dynamically:
            deterministic_model.set_holding_costs(unit_price)
        deterministic_model.set_big_m()
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.model.setParam('TimeLimit', 2 * 60)  # set the time limit to 2 minutes for the gurobi model
        deterministic_model.model.setParam('MIPGap', 0.01)  # set the MIPGap to be 1%

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

    end_time = time.time()  # Stop measuring the time
    runtime = end_time - start_time
    if should_write:
        output_folder = "results"

        output_file = f"simulation_output_p{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_t{n_time_periods}_ep{n_episodes}_S{major_setup_cost}_r{minor_setup_ratio}_beta{beta}_seed{seed}.txt"
        file_path = os.path.join(output_folder, output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        with open(file_path, "a") as f:
            f.write(f"Solution time for this episode is: {runtime} seconds" + "\n")

            for product_index in range(len(products)):
                service_level = sum_fulfilled_demand[product_index] / sum_actual_demand[product_index]
                print(f"Achieved service level for Product {product_index}: {service_level}")
                f.write(f"Achieved service level for Product {product_index}: {service_level}" + "\n")

            f.write(f"Actions for this episode are: {actions}" + "\n")
            f.write(f"Total costs for this episode is: {total_costs}" + "\n")

            f.close()
    print(f"Solution time for this episode is: {runtime} seconds")
    print(actions)

    return total_costs, inventory_levels, start_date, actions, models
