from datetime import timedelta #, datetime

import deterministic_model as det_mod, holt_winters_method #, arima

def simulate(start_date, n_time_periods, products):
    dict_demands = {}
    dict_sds = {}
    # initialize model
    deterministic_model = det_mod.DeterministicModel()

    actions = {}  # Store the first actions for each time step
    inventory_levels = deterministic_model.start_inventory.copy()

    for time in range(n_time_periods):
        deterministic_model = det_mod.DeterministicModel()
        start_date = start_date + timedelta(days=7)

        # Update inventory levels based on previous actions and actual demand
        actual_demands = []
        if time != 0:
            for product_index, product in enumerate(products):
                actual_demand = products[product_index].loc[start_date, "sales_quantity"]
                actual_demands.append(actual_demand)
                added_inventory = max(0,actions[time-1][product_index] - actual_demand)
                inventory_levels[product_index] += added_inventory
            print("Actions at time period ", time-1)
            print(actions[time - 1])
            print("Actual_demand for period ", time-1)
            print(actual_demands)
            print("Inventory levels at start of time period ", time)
            print(inventory_levels)


        for product_index in range(len(products)):
            dict_demands[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)[:n_time_periods][0]
            dict_sds[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)[:n_time_periods][1]
        deterministic_model = det_mod.DeterministicModel()
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.set_safety_stock(dict_sds)

        print("Safety stocks:")
        print(deterministic_model.safety_stock)

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
                    actions[time][product_index] = var.x



    return actions

