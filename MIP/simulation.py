from datetime import timedelta, datetime

from MIP import deterministic_model as det_mod, holt_winters_method, arima

def simulate(start_date, time_periods, products, previous_actions={}):
    dict_demands = {}
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # initialize model
    deterministic_model = det_mod.DeterministicModel()

    first_actions = {}  # Store the first actions for each time step

    for time in range(time_periods):
        start_date = start_date + timedelta(days=7)

        # Update inventory levels based on previous actions and actual demand
        if previous_actions:
            for product_index, product in enumerate(products):
                actual_demand_date = start_date - timedelta(days=7)  # Get the date for actual demand
                actual_demand = products[product_index]["sales_quantity"].get(actual_demand_date.strftime("%Y-%m-%d"), 0)
                deterministic_model.start_inventory[product_index] += previous_actions[product_index] - actual_demand

        for product_index in range(len(products)):
            dict_demands[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.optimize()

        # Extract and store the first action for each product in the current time step
        first_actions[time] = {}
        for var in deterministic_model.model.getVars():
            if var.varName.startswith("ReplenishmentQ"):
                product_index, current_time = map(int, var.varName.split("[")[1].split("]")[0].split(","))
                if current_time == time:
                    first_actions[time][product_index] = var.x
                    print(f"{var.varName}: {var.x:.2f}")

    return first_actions

