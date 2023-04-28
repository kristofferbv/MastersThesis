from datetime import timedelta, datetime

from MIP import deterministic_model as det_mod, holt_winters_method, arima

def simulate(start_date, time_periods, products):
    dict_demands = {}
    # initialize model
    deterministic_model = det_mod.DeterministicModel()

    actions = {}  # Store the first actions for each time step

    for time in range(time_periods):
        start_date = start_date + timedelta(days=7)

        # Update inventory levels based on previous actions and actual demand
        if any(actions):
            for product_index, product in enumerate(products):
                actual_demand = products[product_index].loc[start_date, "sales_quantity"]
                deterministic_model.start_inventory[product_index] += actions[time-1][product_index] - actual_demand

        for product_index in range(len(products)):
            dict_demands[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.optimize()

        # Extract and store the first action for each product in the current time step
        actions[time] = {}
        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                # Only looking at the action at time t = 1, since that is the actual action for this period
                if current_time == 1:
                    actions[time][product_index] = var.x
                    print(f"{var.varName}: {var.x:.2f}")

    return actions

