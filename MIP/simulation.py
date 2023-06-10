from datetime import timedelta, datetime
import pandas as pd
import os
import sys
import deterministic_model as det_mod
from MIP.analysis.analyse_data import plot_sales_quantity
from config_utils import load_config
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
            generated_products += generate_seasonal_data_based_on_products(real_products[last_index:current_index], simulation_length + start_index + n_time_periods + 52)
            last_index = current_index
        # plot_sales_quantity(generated_products)
    start_date = generated_products[0].index[start_index]
    plot_sales_quantity(generated_products)

    sample_data(file_path, start_date, n_time_periods, generated_products, simulation_length)

def sample_data(file_path, start_date, n_time_periods, products, episode_length):
    config = load_config("../config.yml")
    should_set_holding_cost_dynamically = config["simulation"]["should_set_holding_cost_dynamically"]
    if should_set_holding_cost_dynamically:
        unit_price = [df.iloc[0]['average_unit_price'] for df in products]
    dict_demands = {}
    dict_actions = {}  # Store the first actions for each time step
    dict_inventory_levels = {}

    list_dict_historical_demands = []
    list_dict_actions = []
    list_dict_inventory_levels = []
    inventory_levels = [0 for i in range(len(products))]

    for time_step in range(episode_length):
        print(f"Time step {time_step}/{episode_length}")
        actual_demands = []
        for product_index in range(len(products)):
            zero_data = pd.Series([0])
            sales_quantity_data = products[product_index].loc[start_date + pd.DateOffset(weeks=1):start_date + pd.DateOffset(weeks=13), "sales_quantity"]
            sales_quantity_data = pd.concat([zero_data, sales_quantity_data]).reset_index(drop=True)
            dict_demands[product_index] = sales_quantity_data
        deterministic_model = det_mod.DeterministicModel(len(products))
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
        dict_actions = {}
        dict_inventory_levels = {}
        dict_demands = {}
        hei =deterministic_model.model.getVars()
        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                # Only looking at the action at time t = 1, since that is the actual action for this period
                if product_index not in dict_actions.keys():
                    dict_actions[product_index] = []
                # only consider the 13 first actions
                if current_time < 13:
                    if current_time > 0:
                        if var.x < threshold:
                            dict_actions[product_index].append(0)
                        else:
                            dict_actions[product_index].append(var.x)
                else:
                    break
        dict_actions = transform_dictionary(dict_actions)
        list_dict_actions.append(dict_actions)
        list_dict_inventory_levels.append(dict_inventory_levels)
        dict_historic_demands = {}
        dict_inventory_levels[0] = inventory_levels
        period = 1
        current_index = -1
        for product_index in dict_actions.keys():
            if current_index != product_index:
                period = 1
            start_date += timedelta(7)
            for product_index, product in enumerate(products):
                if isinstance(product, pd.DataFrame):
                    demand = products[product_index].loc[start_date, "sales_quantity"]
                else:
                    demand = products[product_index].loc[start_date]

                actual_demands.append(demand)
                previous_il = inventory_levels[product_index]
                inventory_levels[product_index] = max(0, previous_il + dict_actions[product_index][period] - demand)
            dict_inventory_levels[period] =  inventory_levels
        list_dict_inventory_levels.append(dict_inventory_levels)
        list_dict_actions.append(dict_actions)
        for product_index, product in enumerate(products):
            #sample historical data for the last 12 weeks
            zero_data = pd.Series([0])
            historic_sales_quantity_data = product.loc[start_date + pd.DateOffset(weeks=-11):start_date + pd.DateOffset(weeks=14), "sales_quantity"]
            historic_sales_quantity_data = pd.concat([zero_data, historic_sales_quantity_data]).reset_index(drop=True)
            dict_historic_demands[product_index] = historic_sales_quantity_data
        list_dict_historical_demands.append(dict_historic_demands)
        start_date += timedelta(weeks = 13)


        with open(file_path, "a") as f:
            f.write(f"Data for period {time_step}:" + "\n")
            f.write(f"Actions : {list_dict_actions}" + "\n")
            f.write(f"Inventory: {list_dict_inventory_levels}" + "\n")
            f.write(f"Demand: {list_dict_historical_demands}" + "\n")
        f.close()

def transform_dictionary(original_dict):
    new_dict = {}
    for key1, inner_dict in original_dict.items():
        for key2, value in inner_dict.items():
            if key2 not in new_dict:
                new_dict[key2] = {}
            new_dict[key2][key1] = value
    return new_dict


