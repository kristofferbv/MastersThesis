import gurobipy as gp
from gurobipy import GRB
from MIP.config_utils import load_config


class DeterministicModel:
    def __init__(self):

        config = load_config("MIP/config.yml")
        self.n_time_periods = config["n_time_periods"]  # number of time periods
        self.n_products = config["n_products"]
        self.products = [i for i in range(0, self.n_products)]
        self.time_periods = [i for i in range(0, self.n_time_periods + 1)]
        self.tau_periods = [i for i in range(1, self.n_time_periods + 1)]

        # Parameters
        self.major_setup_cost = config["joint_setup_cost"]
        self.minor_setup_cost = config["minor_setup_cost"]
        self.demand_forecast = config["demand_forecast"]
        self.holding_cost = config["holding_cost"]
        self.safety_stock = 0
        self.big_m = config["big_m"]
        self.model = gp.Model('Inventory Control 1')
        self.start_inventory = [0,0,0,0,0,0]
        self.has_been_set_up = False

    def set_demand_forecast(self, demand_forcast):
        self.demand_forecast = demand_forcast

    def reset_model(self):
        self.model.reset(0)
        self.has_been_set_up = False
        self.set_up_model()

    def optimize(self):
        self.model.optimize()

    def set_inventory_levels(self, inventory_levels):
        self.start_inventory = inventory_levels


    def set_up_model(self):
        if self.has_been_set_up:
            return
        replenishment_q = self.model.addVars(self.products, self.time_periods, lb=0, name="ReplenishmentQ")
        order_product = self.model.addVars(self.products, self.time_periods, self.tau_periods, vtype=GRB.BINARY, name="OrderProduct")
        place_order = self.model.addVars(self.time_periods, vtype=GRB.BINARY, name="PlaceOrder")
        inventory_level = self.model.addVars(self.products, self.time_periods, lb=self.safety_stock, name="InventoryLevel")

        # constraints
        # start inventory constraint
        # this will start as a parameter
        start_inventory = self.model.addConstrs((inventory_level[product, self.time_periods[0]] == self.start_inventory[product]) for product in self.products)
        inventory_balance = self.model.addConstrs((inventory_level[product, self.time_periods[i - 1]] + replenishment_q[product, self.time_periods[i]] == self.demand_forecast[product][self.time_periods[i - 1]] + inventory_level[product, self.time_periods[i]] for product in self.products for i in range(1, len(self.time_periods))), name="InventoryBalance")
        minor_setup_incur = self.model.addConstrs((replenishment_q[product, time_period] <= self.big_m[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods), name="MinorSetupIncur")
        major_setup_incur = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for product in self.products for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= place_order[time_period] * self.n_products for time_period in self.time_periods), name="MajorSetupIncur")
        max_one_order = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= 1 for product in self.products for time_period in self.time_periods), name="MaxOneOrder")
        if self.safety_stock == 0:
            minimum_inventory = self.model.addConstrs((inventory_level[product, time_period] >= gp.quicksum(
                order_product[product, time_period, tau_period] * (gp.quicksum(self.demand_forecast[product][time_period + x] for x in range(1, tau_period))) for tau_period in self.tau_periods[:len(self.tau_periods) - time_period])
                                                       for product
                                                       in self.products for time_period in self.time_periods), name="minimumInventory")
        else:
            minimum_inventory = self.model.addConstrs((inventory_level[product, time_period] >= (1 - gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period])) * self.safety_stock[product, time_period, 1] + gp.quicksum(
                order_product[product, time_period, tau_period] * (self.safety_stock[product, time_period, tau_period] + gp.quicksum(self.demand_forecast[(product, time_period + x)] for x in range(1, tau_period))) for tau_period in self.tau_periods[:len(self.tau_periods) - time_period])
                                                       for product
                                                       in self.products for time_period in self.time_periods), name="minimumInventory")

        # objective function
        obj = gp.quicksum(self.major_setup_cost * place_order[time_period] for time_period in self.time_periods) + gp.quicksum(self.minor_setup_cost[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods) + gp.quicksum(
            self.holding_cost[product] * inventory_level[product, time_period] for product in self.products for time_period in self.time_periods)

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.has_been_set_up = True
