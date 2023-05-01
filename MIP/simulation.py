from datetime import timedelta #, datetime

import deterministic_model as det_mod, holt_winters_method #, arima

def simulate(start_date, n_time_periods, products):
    dict_demands = {}
    dict_sds = {}
    # initialize model
    deterministic_model = det_mod.DeterministicModel()

    actions = {}  # Store the first actions for each time step
    inventory_levels = deterministic_model.start_inventory.copy()

    total_costs = 0

    for time in range(n_time_periods):
        deterministic_model = det_mod.DeterministicModel()
        start_date = start_date + timedelta(days=7)

        period_costs = 0

        # Update inventory levels based on previous actions and actual demand
        actual_demands = []
        if time != 0:
            for product_index, product in enumerate(products):
                actual_demand = products[product_index].loc[start_date, "sales_quantity"]
                actual_demands.append(actual_demand)
                added_inventory = max(0,actions[time-1][product_index] - actual_demand)  # skal vi ikke her ta at inventory level skal være max av 0 of invnetory level før + actions - actual demand??

                #add holding costs or shortage costs
                if inventory_levels[product_index] + actions[time-1][product_index] - actual_demand > 0:
                    period_costs += (inventory_levels[product_index] + actions[time-1][product_index] - actual_demand)* deterministic_model.holding_cost[product_index]
                else:
                    period_costs += abs(inventory_levels[product_index] + actions[time-1][product_index] - actual_demand) * deterministic_model.shortage_cost[product_index]
                
                major_setup_added = False

                #add setup costs:
                if actions[time-1][product_index] > 0:
                    period_costs += deterministic_model.minor_setup_cost[product_index]

                    #to only add major setup costs once if an order is made
                    if major_setup_added == False:
                        period_costs += deterministic_model.major_setup_cost
                        major_setup_added = True

                inventory_levels[product_index] += added_inventory

            print("Period costs: ")
            print(period_costs)

            total_costs += period_costs    


            print("Actions at time period ", time-1)
            print(actions[time - 1])
            print("Actual_demand for period ", time-1)
            print(actual_demands)
            print("Inventory levels at start of time period ", time)
            print(inventory_levels)

            print("Total costs at time peirod : ", time)
            print(total_costs)

        for product_index in range(len(products)):
            dict_demands[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)[:n_time_periods][0]
            dict_sds[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)[:n_time_periods][1]
        deterministic_model = det_mod.DeterministicModel()
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.set_safety_stock(dict_sds)

        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.set_inventory_levels(inventory_levels)
        deterministic_model.set_up_model()
        deterministic_model.optimize()

        # Extract and store the first action for each product in the current time step
        actions[time] = {}
        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                # Only looking at the action at time t = 1, since that is the actual action for this period
                if current_time == 1:
                    print(var)
                    actions[time][product_index] = var.x



    return actions

