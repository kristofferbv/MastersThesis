from datetime import timedelta, datetime

from MIP import deterministic_model as det_mod, holt_winters_method


def simulate(start_date, time_periods, products):
    dict_demands = {}
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    # initialize model
    deterministic_model = det_mod.DeterministicModel()
    for time in range(time_periods):
        start_date = start_date + timedelta(days=7)
        # TODO:  Should set new inventory level here!
        for product_index in range(len(products)):
            dict_demands[product_index] = holt_winters_method.forecast(products[product_index]["sales_quantity"], start_date)
        deterministic_model.set_demand_forecast(dict_demands)
        deterministic_model.model.setParam("OutputFlag", 0)
        deterministic_model.optimize()
        for var in deterministic_model.model.getVars():
            print(f"{var.varName}: {var.x:.2f}")