import gurobipy as gp
from gurobipy import GRB
from config_utils import load_config
from scipy.stats import norm
import numpy as np

import math


class DeterministicModel:
    def __init__(self):

        config = load_config("config.yml")
        self.n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
        self.n_products = config["deterministic_model"]["n_products"]  # number of product types
        self.products = [i for i in range(0, self.n_products)]
        self.time_periods = [i for i in range(0, self.n_time_periods + 1)]
        self.tau_periods = [i for i in range(0, self.n_time_periods + 1)]

        # Parameters
        self.major_setup_cost = config["deterministic_model"]["joint_setup_cost"]
        self.minor_setup_cost = config["deterministic_model"]["minor_setup_cost"]
        self.holding_cost = config["deterministic_model"]["holding_cost"]
        self.safety_stock = {}
        self.big_m = config["deterministic_model"]["big_m"]
        self.model = gp.Model('Inventory Control 1')
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.has_been_set_up = False
        self.service_level = {}
        self.service_level = config["deterministic_model"]["service_level"]
        self.shortage_cost = config["deterministic_model"]["shortage_cost"]
        self.should_include_safety_stock = config["deterministic_model"]["should_include_safety_stock"]

        # change shortage cost based on formula 
        # could make an if sentence if this could be set by the user
        for product_index in range(self.n_products):
            self.shortage_cost[product_index] = self.holding_cost[product_index]/(1/self.service_level[product_index] - 1)

        #for product_index in range(self.n_products):
         #   self.service_level[product_index][0] = config["deterministic_model"]["service_level"][product_index]
          #  for period in range(1, self.tau_periods):
           #     self.service_level[product_index][period] = config["deterministic_model"]["service_level"][product_index]


    def set_demand_forecast(self, demand_forecast):
        self.demand_forecast = demand_forecast
        self.model.update()

    def set_safety_stock(self, standard_deviations):
        for product_index in range(self.n_products):
            self.safety_stock[product_index] = norm.ppf(self.service_level[product_index]) * np.array(standard_deviations[product_index])
        self.model.update()

    def reset_model(self):
        self.model.reset(0)
        # self.has_been_set_up = False
        # self.set_up_model()

    def optimize(self):
        self.model.optimize()

    def set_inventory_levels(self, inventory_levels):
        self.start_inventory = inventory_levels
        self.model.update()

    def set_up_model(self):
        if self.has_been_set_up:
            return
        replenishment_q = self.model.addVars(self.products, self.time_periods, lb=0, name="ReplenishmentQ")
        order_product = self.model.addVars(self.products, self.time_periods, self.tau_periods, vtype=GRB.BINARY, name="OrderProduct")
        place_order = self.model.addVars(self.time_periods, vtype=GRB.BINARY, name="PlaceOrder")
        # had lb= safety stock - but will not be valid for t=0, make sure with constraints later lower bound
        inventory_level = self.model.addVars(self.products, self.time_periods, lb=0, name="InventoryLevel")

        start_inventory = self.model.addConstrs((inventory_level[product, self.time_periods[0]] == self.start_inventory[product]) for product in self.products)
        inventory_balance = self.model.addConstrs((inventory_level[product, self.time_periods[i - 1]] + replenishment_q[product, self.time_periods[i]] == self.demand_forecast[product][self.time_periods[i]] + inventory_level[product, self.time_periods[i]] for product in self.products for i in range(1, len(self.time_periods))), name="InventoryBalance")
        minor_setup_incur = self.model.addConstrs((replenishment_q[product, time_period] <= self.big_m[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods), name="MinorSetupIncur")
        major_setup_incur = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for product in self.products for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= place_order[time_period] * self.n_products for time_period in self.time_periods), name="MajorSetupIncur")
        max_one_order = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= 1 for product in self.products for time_period in self.time_periods), name="MaxOneOrder")
        if self.should_include_safety_stock:
            #minimum_inventory = self.model.addConstrs((inventory_level[product, time_period] >= 0 for product in self.products for time_period in self.time_periods[1:]), name="minimumInventory")
            minimum_inventory = self.model.addConstrs((inventory_level[product, self.time_periods[i]] >= gp.quicksum(order_product[product, self.time_periods[i], self.tau_periods[j]] for j in range(0,2)) *  self.safety_stock[product][i] +
                                                       gp.quicksum(order_product[product, self.time_periods[i], self.tau_periods[j]] * (self.safety_stock[product][i]) for j in (2, len(self.time_periods)-i))
                                                    for product in self.products for i in range(1, len(self.time_periods))), name="minimumInventory")
            
            
            #minimum_inventory = self.model.addConstrs((inventory_level[product, self.time_periods[i]] >= self.safety_stock[product][i] for product in self.products for i in range(1, len(self.time_periods))), name="minimumInventory")
              # #+ gp.quicksum(
                #order_product[product, time_period, tau_period] * (self.safety_stock[product] #+ gp.quicksum(self.demand_forecast[(product, time_period + x)] for x in range(1, tau_period))) for tau_period in self.tau_periods[2:len(self.tau_periods) - time_period])
               
        else:
            minimum_inventory = self.model.addConstrs((inventory_level[product, time_period] >= 0 for product in self.products for time_period in self.time_periods[1:]), name="minimumInventory")
            
        # objective function
        obj = gp.quicksum(self.major_setup_cost * place_order[time_period] for time_period in self.time_periods) + gp.quicksum(self.minor_setup_cost[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods) + gp.quicksum(
            self.holding_cost[product] * inventory_level[product, time_period] for product in self.products for time_period in self.time_periods)

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.has_been_set_up = True
