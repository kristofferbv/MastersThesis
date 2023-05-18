from datetime import timedelta
import deterministic_model as det_mod
import sarima
import holt_winters_method
from config_utils import load_config
import numpy as np


def simulate(start_date, n_time_periods, products):
    config = load_config("config.yml")
    forecasting_method = config["simulation"]["forecasting_method"]  # number of time periods
    verbose = config["simulation"]["verbose"]  # number of time periods
    should_set_holding_cost_dynamically = config["simulation"]["should_set_holding_cost_dynamically"]
    if should_set_holding_cost_dynamically:
        unit_costs = [df.iloc[0]['average_unit_cost'] for df in products]

    dict_demands = {}
    dict_sds = {}
    # initialize model

    actions = {}  # Store the first actions for each time step
    orders = {}
    inventory_levels = [0 for i in range(len(products))]

    total_costs = 0

    shortage_costs = 0
    holding_costs = 0
    setup_costs = 0

    avg_items = 0

    actual_demand_product = {}

    for time in range(n_time_periods):
        start_date = start_date + timedelta(days=7)

        period_costs = 0
        major_setup_added = False

        # Update inventory levels based on previous actions and actual demand
        actual_demands = []
        items_ordered = 0

        if time != 0:
            for product_index, product in enumerate(products):
                actual_demand = products[product_index].loc[start_date, "sales_quantity"]
                actual_demands.append(actual_demand)
                actual_demand_product[time] = actual_demands
                #print("inventory levels at the beginning of period ", time)
                #print(inventory_levels)

                # added_inventory = max(0,actions[time-1][product_index] - actual_demand)  # skal vi ikke her ta at inventory level skal være max av 0 of invnetory level før + actions - actual demand??

                # add holding costs or shortage costs
                if inventory_levels[product_index] + actions[time - 1][product_index] - actual_demand > 0:
                    holding_costs += (inventory_levels[product_index] + actions[time - 1][product_index] - actual_demand) * deterministic_model.holding_cost[product_index]
                    period_costs += (inventory_levels[product_index] + actions[time - 1][product_index] - actual_demand) * deterministic_model.holding_cost[product_index]
                else:
                    shortage_costs += abs(inventory_levels[product_index] + actions[time - 1][product_index] - actual_demand) * deterministic_model.shortage_cost[product_index]
                    period_costs += abs(inventory_levels[product_index] + actions[time - 1][product_index] - actual_demand) * deterministic_model.shortage_cost[product_index]

                # add setup costs:
                if actions[time - 1][product_index] > 0:
                    setup_costs += deterministic_model.minor_setup_cost[product_index]
                    period_costs += deterministic_model.minor_setup_cost[product_index]
                    items_ordered += 1

                    # to only add major setup costs once if an order is made
                    if not major_setup_added:
                        setup_costs += deterministic_model.major_setup_cost
                        period_costs += deterministic_model.major_setup_cost
                        major_setup_added = True

                previous_il = inventory_levels[product_index]
                inventory_levels[product_index] = max(0, previous_il + actions[time - 1][product_index] - actual_demand)

            total_costs += period_costs
            avg_items += items_ordered
            if verbose:
                print("Period costs: ")
                print(period_costs)

                print("Actions at time period ", time - 1)
                print(actions[time - 1])

                print("Actual_demand for period ", time - 1)
                print(actual_demands)

                print("Inventory levels at start of time period ", time)
                print(inventory_levels)

                print("Total costs at time period : ", time)
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

        deterministic_model = det_mod.DeterministicModel()
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
        actions[time] = {}
        threshold = 1e-10

        orders[time] = {}

        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                # Only looking at the action at time t = 1, since that is the actual action for this period
                if current_time == 1:
                    actions[time][product_index] = var.x
                    if abs(actions[time][product_index]) < threshold:
                        actions[time][product_index] = 0

            if var.varName.startswith("OrderProduct"):
                for tau in deterministic_model.tau_periods:
                    product_index, current_time, tau = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                    # Only looking at the action at time t = 1, since that is the actual action for this period
                    if current_time == 1:
                        if product_index not in orders[time]:
                            orders[time][product_index] = {}
                        orders[time][product_index][tau] = var.x
                        if abs(orders[time][product_index][tau]) < threshold:
                            orders[time][product_index][tau] = 0

   
    print("Total costs at after all periods : ")
    print(total_costs)
    print("Total shortage costs")
    print(shortage_costs)
    print("Holding costs:")
    print(holding_costs)
    print("Setup costs")
    print(setup_costs)
    print(actions)
    #print("orders")
    #print(orders)
    runtime = deterministic_model.model.Runtime
    print("The run time is %f" % runtime)

    print("Average items ordered is:")
    print(avg_items/n_time_periods)

    average_demands = []
    for product_index in range(len(products)):
        tot_demand = 0
        for time_period in range(1, n_time_periods):
            tot_demand += actual_demand_product[time_period][product_index]

        avg_demand = tot_demand/n_time_periods
        average_demands.append(avg_demand*52)

    print("Average demands are:")
    print(average_demands)

    for product_index in range(len(products)):
        set_up = deterministic_model.major_setup_cost/avg_items + deterministic_model.minor_setup_cost[product_index]

        d1 = (2*set_up)/deterministic_model.holding_cost[product_index]*(1**2)
        q1 = np.sqrt((2*set_up*d1)/(deterministic_model.holding_cost[product_index]))
        set_up_per_unit_1 = set_up / q1

        print("EOQ cost of ordering product" ,product_index, " every time period:", str(set_up_per_unit_1))

        d2 = (2*set_up)/(deterministic_model.holding_cost[product_index]*(2**2))
        q2 = np.sqrt((2*set_up*d2)/deterministic_model.holding_cost[product_index])
        set_up_per_unit_2 = set_up / q2

        print("EOQ cost of ordering product" ,product_index, "every second time period:", set_up_per_unit_2)

        print("Service level for increasing tau from 1 to 2 is:" )
        service_level_1_2 = (set_up_per_unit_2-set_up_per_unit_1)/(set_up_per_unit_2-set_up_per_unit_1+deterministic_model.holding_cost[product_index])
        print(service_level_1_2)

        d3 = (2*set_up)/(deterministic_model.holding_cost[product_index]*(3**2))
        q3 = np.sqrt((2*set_up*d3)/deterministic_model.holding_cost[product_index])
        set_up_per_unit_3 = set_up / q3

        print("EOQ cost of ordering product" ,product_index, "every third time period:", set_up_per_unit_3)

        print("Service level for increasing tau from 2 to 3 is:" )
        service_level_2_3 = (set_up_per_unit_3-set_up_per_unit_2)/(set_up_per_unit_3-set_up_per_unit_2+deterministic_model.holding_cost[product_index])
        print(service_level_2_3)

        

        print("")

        #print("EOQ cost of ordering every third time period:")


        #print("EOQ cost of ordering every fourth time period:")







    



    return actions
