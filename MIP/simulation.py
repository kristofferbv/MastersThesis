import pickle
from datetime import timedelta

import pandas as pd
import os
import numpy as np
import sys
import time
import statistics
from scipy import stats

import deterministic_model as det_mod
from MIP.analysis.analyse_data import plot_sales_quantity
from MIP.forecasting import holt_winters_method, sarima
from MIP.standard_deviation import get_initial_std_dev, get_std_dev
from config_utils import load_config
import generate_data


def simulate(real_products, config, beta=None, n_time_periods=None, total_costs_best_beta=None, best_beta=None):
    if n_time_periods is None:
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
    if beta is None:
        beta = config["deterministic_model"]["beta"]

    if best_beta is None:
        best_beta = beta

    output_folder = "results"
    output_file = f"simulation_output_p{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_t{n_time_periods}_ep{n_episodes}_S{major_setup_cost}_r{minor_setup_ratio}_beta{beta}_seed{seed}.txt"
    file_path = os.path.join(output_folder, output_file)

    if os.path.exists(file_path):
        os.remove(file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data_to_write = []

    generated_products = []
    current_index = 0
    last_index = 0
    # with open(file_path, "a") as f:
    generation_length = (simulation_length + reset_length) * n_episodes + (warm_up_length * should_perform_warm_up) + start_index + n_time_periods + 52
    for category in product_categories.keys():
        number_of_products = product_categories[category]
        current_index += number_of_products
        if category == "erratic":
            generated_products += generate_data.generate_seasonal_data_for_erratic_demand(real_products[last_index:current_index], generation_length, seed)
        elif category == "smooth":
            generated_products += generate_data.generate_seasonal_data_for_smooth_demand(real_products[last_index:current_index], generation_length, seed)

        else:
            generated_products += generate_data.generate_seasonal_data_for_intermittent_demand(real_products[last_index:current_index], generation_length, seed)
        last_index = current_index
    # plot_sales_quantity(generated_products)

    data_to_write.append("Number of products from each cateogry is:  Erratic: " + str(product_categories["erratic"]) + ", Smooth: " + str(product_categories["smooth"]) + ", Intermittent: " + str(product_categories["intermittent"]) + ", Lumpy: " + str(product_categories["lumpy"]))

    total_costs = []
    list_mean = []
    list_sem = []
    list_std_run_time = []
    list_avg_forecast_errors = []
    list_std_forecast_errors = []
    list_avg_optimality_gap = []
    list_std_optimality_gap = []
    service_levels = {product_index: [] for product_index in range(n_products)}
    # actual_demand_list =  {episode: [] for episode in range(n_episodes)}

    inventory_levels = None
    start_date = generated_products[0].index[start_index]
    end_date = generated_products[0].index[generation_length - 13]
    all_forecasts = {}
    all_models = {}
    all_std_devs = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')



    if check_if_os_path_exists(n_products, n_erratic, n_smooth, n_intermittent, n_lumpy, seed):
        with open(f'forecast_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'rb') as f:
            all_forecasts = pickle.load(f)
        with open(f'models_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'rb') as f:
            all_models = pickle.load(f)
        with open(f'std_devs_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'rb') as f:
            all_std_devs = pickle.load(f)
    else:
        all_forecasts, all_models, all_std_devs = precalculate_forecasts(date_range, n_time_periods, generated_products, config, n_products, n_erratic, n_smooth, n_intermittent, n_lumpy, seed)

    models = {}
    if should_perform_warm_up and warm_up_length > 0:
        print("Warming up")
        inventory_levels, start_date, models = perform_warm_up(generated_products, start_date, n_time_periods, config, all_forecasts, all_std_devs)

    avg_run_time = 0


    
    


    for episode in range(n_episodes):
        # simulate and sample costs
        print(f"Running simulation episode {episode}")

        data_to_write.append(f"Start inventory levels of episode: {episode} are {inventory_levels}")

        costs, inventory_levels, start_date, actions, tau_values, models, avg_run_time_time_step, std_run_time, service_level, actual_demands, avg_forecast_errors, std_forecast_errors, avg_optimality_gap, std_optimality_gap = run_one_episode(start_date, n_time_periods, generated_products, simulation_length, config, all_forecasts, all_std_devs, models=models, inventory_levels=inventory_levels, beta=beta)
        print(f"Total costs for episode {episode} was: ", costs)
        total_costs.append(costs)
        list_mean.append(sum(total_costs) / len(total_costs))
        list_std_run_time.append(std_run_time)
        list_avg_forecast_errors.append(avg_forecast_errors)
        list_std_forecast_errors.append(std_forecast_errors)
        avg_run_time += avg_run_time_time_step
        list_avg_optimality_gap.append(avg_optimality_gap)
        list_std_optimality_gap.append(std_optimality_gap)
        if episode > 0:
            list_sem.append(stats.sem(total_costs))
        data_to_write.append(f"Actions for episode {episode} are: {actions}")
        data_to_write.append(f"Tau values for episode {episode} are: {tau_values}")
        # data_to_write.append(f"Costs for episode {episode} is: {costs}")
        # f.write(f"Actions for episode {episode} are: {actions}" + "\n")
        # print(f"Actions for episode {episode} are: {actions}")
        # print(service_level)
        data_to_write.append(f"Actual demands for episode {episode} are: {actual_demands}")

        for product_index in range(n_products):
            service_levels[product_index].append(service_level[product_index])

        if reset_length > 0:
            print("Resetting...")
            # resetting
            costs, inventory_levels, start_date, _, _, models, _, _, _, _, _, _, _, _ = run_one_episode(start_date, n_time_periods, generated_products, reset_length, config, all_forecasts, all_std_devs, models=models, inventory_levels=inventory_levels)

        avg_run_time = avg_run_time / n_episodes

        '''
        # check first stopping criterion
        if episode > 20 and (stats.sem(total_costs) * 2 * 1.96 < 0.02 * np.mean(total_costs)):
            print(f"Stopping early after {episode} episodes")
            break
        '''
        # Check the stopping criterion
        if episode > 20 and len(total_costs) > 0 and total_costs_best_beta is not None:

            # Calculate differences
            diffs = np.array(total_costs) - np.array(total_costs_best_beta[:len(total_costs)])

            # Calculate mean of the differences
            mean_diff = np.mean(diffs)

            # Calculate standard error of the differences
            se_diff = stats.sem(diffs)

            # Define confidence interval (90% in this case)
            ci = 0.90

            # Degrees of freedom
            df = len(diffs) - 1

            # Calculate the confidence interval
            ci_low, ci_high = stats.t.interval(ci, df, loc=mean_diff, scale=se_diff)

            print(f"Confidence interval is: [{ci_low}, {ci_high}]")
            data_to_write.append(f"Confidence interval is: [{ci_low}, {ci_high}]")

            # Check if confidence interval contains zero
            if ci_low > 0:
                print(f"Beta value {best_beta} is significantly better than beta value {beta}")
                data_to_write.append(f"Beta value {best_beta} is significantly better than beta value {beta}")
                break
            elif episode == n_episodes-1:
                mean_beta = np.mean(total_costs)
                mean_best_beta = np.mean(total_costs_best_beta)
                if ci_high < 0:
                    print(f"Beta value {beta} is significantly better than beta value {best_beta}")
                    data_to_write.append(f"Beta value {beta} is significantly better than beta value {best_beta}")
                    best_beta = beta
                    total_costs_best_beta = total_costs

                elif mean_beta < mean_best_beta:
                    best_beta = beta
                    total_costs_best_beta = total_costs
                    print(f"No beta value is significantly better than the other, but Beta value {beta} has a lower mean value {best_beta}")
                    data_to_write.append(f"No beta value is significantly better than the other, but Beta value {beta} has a lower mean value {best_beta}")
                else:
                    print(f"No beta value is significantly better than the other, but Beta value {best_beta} has a lower mean value {beta}")
                    data_to_write.append(f"No beta value is significantly better than the other, but Beta value {best_beta} has a lower mean value {beta}")
                    

            '''
            # Calculate the variance of the difference
            var_diff = np.var(total_costs_beta1) / len(total_costs_beta1) + np.var(total_costs_beta2) / len(total_costs_beta2)

            # Calculate the standard error of the difference
            sem_diff = np.sqrt(var_diff)

            # Calculate the confidence interval
            conf_interval = stats.t.interval(0.95, df=min(len(total_costs_beta1), len(total_costs_beta2))-1, loc=np.mean(total_costs_beta1) - np.mean(total_costs_beta2), scale=sem_diff)

            # Check if the confidence interval does not include zero
            if conf_interval[0] > 0 or conf_interval[1] < 0:
                print(f"Beta value {beta_value1} is significantly better than beta value {beta_value2}")
                break
            '''
            

    if should_write:
        data_to_write.append(f"Total costs for each period are: {total_costs}")
        # data_to_write.append(f"Total average costs for all episodes is: {sum(total_costs) / len(total_costs)}")
        data_to_write.append(f'List of mean costs for each period: {list_mean}')
        data_to_write.append(f'List of standard error of mean of total costs for each period: {list_sem}')
        data_to_write.append(f"Service levels are: {service_levels}")
        # data_to_write.append(f"Actual demands are: {actual_demand_list}")

        standard_deviation_costs = statistics.stdev(total_costs)
        data_to_write.append(f"Standard deviations of costs: {standard_deviation_costs}")

        data_to_write.append(f"The average time to run the model in Gurobi is: {avg_run_time}")
        data_to_write.append(f"The standard deviations of run time in Gurobi for each epiode is : {list_std_run_time}")

        data_to_write.append(f"The average optimality gap in Gurobi for each episode is: {list_avg_optimality_gap}")  # kan ha et tall overall her?
        data_to_write.append(f"The standard deviations of optimality gap in Gurobi for each epiode is : {list_std_optimality_gap}")

        data_to_write.append(f"The average forecasting errors are : {list_avg_forecast_errors}")
        data_to_write.append(f"The standard deviations forecasting errors for each episode are : {list_std_forecast_errors}")

        data_string = "\n".join(data_to_write)
        with open(file_path, 'a', buffering=8192) as f:
            f.write(data_string + "\n")
        f.close()

        print(f"Total average costs for all episodes is: {sum(total_costs) / len(total_costs)}")
        print(f"Best beta value is {best_beta}")

        if total_costs_best_beta is None:
            total_costs_best_beta = total_costs

        
        return total_costs_best_beta, best_beta


def perform_warm_up(products, start_date, n_time_periods, config, all_forecasts, all_std_devs):
    warm_up_length = config["simulation"]["warm_up_length"]  # This is the number of time periods we are using to warm up
    inventory_levels = [0 for i in range(len(products))]
    # for i in range(simulation_length):
    _, inventory_levels, start_date, _, _, models, _, _, _, _, _, _, _, _ = run_one_episode(start_date, n_time_periods, products, warm_up_length, config, all_forecasts, all_std_devs, inventory_levels=inventory_levels)
    return inventory_levels, start_date, models


def run_one_episode(start_date, n_time_periods, products, episode_length, config, all_forecasts, all_std_devs, models=None, inventory_levels=None, beta=None):
    n_episodes = config["simulation"]["n_episodes"]  # This is the number of times we run a full simulation
    should_write = config["simulation"]["should_write"]
    product_categories = config["deterministic_model"]["product_categories"]
    seed = config["main"]["seed"]
    n_products = sum(product_categories.values())
    n_erratic = product_categories["erratic"]
    n_smooth = product_categories["smooth"]
    n_intermittent = product_categories["intermittent"]
    n_lumpy = product_categories["lumpy"]
    major_setup_cost = config["deterministic_model"]["joint_setup_cost"]
    minor_setup_ratio = config["deterministic_model"]["minor_setup_ratio"]
    if beta is None:
        beta = config["deterministic_model"]["beta"]

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
    actual_demands = {}
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

    run_time_list = []
    gap_list = []

    for time_step in range(episode_length):
        print(f"Time step {time_step}/{episode_length}")
        start_date = start_date + timedelta(days=7)

        period_costs = 0
        major_setup_added = False
        # Update inventory levels based on previous actions and actual demand
        actual_demands[time_step] = {}
        forecasts = all_forecasts[str(start_date.date())]
        std_devs = all_std_devs[str(start_date.date())]
        for product_index, product in enumerate(products):
            if time_step != 0:
                if isinstance(product, pd.DataFrame):
                    demand = products[product_index].loc[start_date, "sales_quantity"]
                else:
                    demand = products[product_index].loc[start_date]
                forecast_errors[product_index] = abs(demand - prev_forecast[product_index])

                actual_demands[time_step][product_index] = demand

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

            if forecasting_method == "holt_winter":
                if time_step == 0:

                    dict_demands[product_index], train = holt_winters_method.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
                    dict_sds[product_index] = get_initial_std_dev(train, n_time_periods)

                    # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                    prev_std_dev[product_index] = dict_sds[product_index][1]
                    prev_forecast[product_index] = dict_demands[product_index][1]
                else:
                    dict_sds[product_index] = get_std_dev(prev_std_dev[product_index], forecast_errors[product_index], n_time_periods, alpha=0.1)
                    dict_demands[product_index], _ = holt_winters_method.forecast(products[product_index], start_date, n_time_periods=n_time_periods)

                    # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                    prev_std_dev[product_index] = dict_sds[product_index][1]
                    prev_forecast[product_index] = dict_demands[product_index][1]

            elif forecasting_method == "sarima":
                if time_step == 0:
                    # fitting the model
                    dict_demands[product_index], models[product_index], train = sarima.forecast(products[product_index], start_date, n_time_periods=n_time_periods)
                    dict_sds[product_index] = get_initial_std_dev(train, n_time_periods)
                    # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                    prev_std_dev[product_index] = dict_sds[product_index][1]
                    prev_forecast[product_index] = dict_demands[product_index][1]
                else:
                    dict_sds[product_index] = get_std_dev(prev_std_dev[product_index], forecast_errors[product_index], n_time_periods, alpha=0.1)
                    dict_demands[product_index], _, _ = sarima.forecast(products[product_index], start_date, model=models[product_index], n_time_periods=n_time_periods)
                    # storing std dev and forecast to use for updating the std deviation of errors in the forecast
                    prev_std_dev[product_index] = dict_sds[product_index][1]
                    prev_forecast[product_index] = dict_demands[product_index][1]

            else:
                raise ValueError(f"Forecasting method must be either 'sarima' or 'holt_winter', but is: {forecasting_method}")

        total_costs += period_costs
        if verbose:
            print("Period costs: ")
            print(period_costs)

            print("Actions at time period ", time_step - 1)
            print(actions[time_step - 1])

            print("Actual_demand for period ", time_step - 1)
            print(actual_demands[time_step - 1][product_index])

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

        deterministic_model = det_mod.DeterministicModel(len(products), config, beta=beta, n_time_periods=n_time_periods)
        deterministic_model.set_demand_forecast(dict_demands)
        if should_set_holding_cost_dynamically:
            deterministic_model.set_holding_costs(unit_price)
        deterministic_model.set_safety_stock(dict_sds)
        deterministic_model.set_big_m()
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.model.setParam('TimeLimit', 2 * 60)  # set the time limit to 2 minutes for the gurobi model
        # deterministic_model.model.setParam('MIPGap', 0.01)  # set the MIPGap to be 1%

        deterministic_model.set_inventory_levels(inventory_levels)
        deterministic_model.set_up_model()
        deterministic_model.optimize()

        run_time_time_step = deterministic_model.model.Runtime
        run_time_list.append(run_time_time_step)

        gap_step = deterministic_model.model.MIPGap
        gap_list.append(gap_step)

        # Extract and store the first action for each product in the current time step
        actions[time_step] = {}
        threshold = 1e-5

        # Extract the tau values for the first periods
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

    # if should_write:
    # output_folder = "results"

    # output_file = f"simulation_output_p{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_t{n_time_periods}_ep{n_episodes}_S{major_setup_cost}_r{minor_setup_ratio}_beta{beta}_seed{seed}.txt"
    # file_path = os.path.join(output_folder, output_file)
    # if not os.path.exists(output_folder):
    #   os.makedirs(output_folder)

    # with open(file_path, "a") as f:

    service_levels = []

    for product_index in range(n_products):
        if sum_actual_demand[product_index] == 0:
            service_level = 1
        else:
            service_level = sum_fulfilled_demand[product_index] / sum_actual_demand[product_index]
        service_levels.append(service_level)

        # print(f"Achieved service level for Product {product_index}: {service_level}")
        # f.write(f"Achieved service level for Product {product_index}: {service_level}" + "\n")

        # f.write(f"Actions for this episode are: {actions}" + "\n")
        # f.write(f"Total costs for this episode is: {total_costs}" + "\n")

        # f.close()
    # print(actions)

    avg_run_time_time_step = (sum(run_time_list)) / episode_length
    std_run_time = statistics.stdev(run_time_list)

    avg_optimaliy_gap = (sum(gap_list) / len(gap_list))
    std_optimality_gap = statistics.stdev(gap_list)

    avg_forecast_errors = sum(forecast_errors.values()) / len(forecast_errors.values())
    std_forecast_errors = statistics.stdev(forecast_errors.values())

    return total_costs, inventory_levels, start_date, actions, orders, models, avg_run_time_time_step, std_run_time, service_levels, actual_demands, avg_forecast_errors, std_forecast_errors, avg_optimaliy_gap, std_optimality_gap


def precalculate_forecasts(date_range, n_time_periods, products, config, n_products, n_erratic, n_smooth, n_intermittent, n_lumpy, seed):
    forecasting_method = config["simulation"]["forecasting_method"]
    all_forecasts = {}
    all_models = {}
    all_std_devs = {}
    count = 0
    prev_std_dev = {}
    prev_forecast = {}
    for date in date_range:
        forecasts = {}
        models = {}
        std_devs = {}
        forecast_errors = {}

        for product_index, product in enumerate(products):

            if forecasting_method == "holt_winter":

                if count == 0:
                    forecasts[product_index], train = holt_winters_method.forecast(product, date, n_time_periods=n_time_periods)
                    std_devs[product_index] = get_initial_std_dev(train, n_time_periods)
                    prev_std_dev[product_index] = std_devs[product_index][1]
                    prev_forecast[product_index] = forecasts[product_index][1]
                else:
                    demand = products[product_index].loc[date, "sales_quantity"]
                    forecast_errors[product_index] = abs(demand - prev_forecast[product_index])
                    forecasts[product_index], train = holt_winters_method.forecast(product, date, n_time_periods=n_time_periods)
                    std_devs[product_index] = get_std_dev(prev_std_dev[product_index], forecast_errors[product_index], n_time_periods, alpha=0.1)
                    prev_std_dev[product_index] = std_devs[product_index][1]
                    prev_forecast[product_index] = forecasts[product_index][1]

            elif forecasting_method == "sarima":
                if count == 0:
                    forecasts[product_index], train = sarima.forecast(product, date, n_time_periods=n_time_periods)
                    std_devs[product_index] = get_initial_std_dev(train, n_time_periods)
                    prev_std_dev[product_index] = std_devs[product_index][1]
                    prev_forecast[product_index] = forecasts[product_index][1]
                else:
                    demand = products[product_index].loc[date, "sales_quantity"]
                    forecast_errors[product_index] = abs(demand - prev_forecast[product_index])
                    forecasts[product_index], train = sarima.forecast(product, date, n_time_periods=n_time_periods)
                    std_devs[product_index] = get_std_dev(prev_std_dev[product_index], forecast_errors[product_index], n_time_periods, alpha=0.1)
            else:
                raise ValueError(f"Forecasting method must be either 'sarima' or 'holt_winter', but is: {forecasting_method}")
        count += 1

        all_forecasts[str(date.date())] = forecasts
        all_models[str(date.date())] = models
        all_std_devs[str(date.date())] = std_devs

    with open(f'forecast_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'wb') as f:
        pickle.dump(all_forecasts, f)
    with open(f'models_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'wb') as f:
        pickle.dump(all_models, f)
    with open(f'std_devs_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl', 'wb') as f:
        pickle.dump(all_std_devs, f)

    return all_forecasts, all_models, all_std_devs


def check_if_os_path_exists(n_products, n_erratic, n_smooth, n_intermittent, n_lumpy, seed):
    if os.path.exists(f'forecast_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl') and \
            os.path.exists(f'models_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl') and \
            os.path.exists(f'std_devs_sp{n_products}_er{n_erratic}_sm{n_smooth}_in{n_intermittent}_lu{n_lumpy}_seed{seed}.pkl'):
        return True
    return False
