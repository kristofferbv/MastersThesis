import os

import gurobipy as gp
import pandas as pd
from gurobipy import GRB
from keras.saving.saving_api import load_model
from sklearn.preprocessing import StandardScaler

from config_utils import load_config
from scipy.stats import norm
import numpy as np

import math
from collections import defaultdict

from generate_data import generate_seasonal_data_based_on_products


class DeterministicModel:
    def __init__(self, real_products):

        config = load_config("../config.yml")
        self.n_time_periods = config["deterministic_model"]["n_time_periods"]  # number of time periods
        self.n_products = config["deterministic_model"]["n_products"]  # number of product types
        self.products = [i for i in range(0, self.n_products)]
        self.time_periods = [i for i in range(0, self.n_time_periods + 1)]
        self.tau_periods = [i for i in range(1, self.n_time_periods + 1)]

        # Parameters
        self.major_setup_cost = config["deterministic_model"]["joint_setup_cost"]
        self.minor_setup_cost = config["deterministic_model"]["minor_setup_cost"]
        self.holding_cost = config["deterministic_model"]["holding_cost"]
        self.safety_stock = defaultdict(lambda: defaultdict(dict))
        #self.big_m = config["deterministic_model"]["big_m"]
        self.big_m = defaultdict(lambda: defaultdict(dict))
        self.model = gp.Model('Inventory Control 1')
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.has_been_set_up = False
        self.service_level = defaultdict(dict)
        #self.service_level = config["deterministic_model"]["service_level"]
        self.shortage_cost = config["deterministic_model"]["shortage_cost"]
        self.should_include_safety_stock = config["deterministic_model"]["should_include_safety_stock"]
        self.actors = []
        generated_products = generate_seasonal_data_based_on_products(real_products, 500, 15)

        self.scaled_products = self.normalize_demand(generated_products)

        # for i in self.n_products:
        #     loaded_model = load_model(os.path.join('models', f'actor_model_{i}'))
        #     self.actors.append(loaded_model)

        # change shortage cost based on formula 
        # could make an if sentence if this could be set by the user
        # for product_index in range(self.n_products):
         #   self.shortage_cost[product_index] = self.holding_cost[product_index]/(1/self.service_level[product_index] - 1)

        service_levels = config["deterministic_model"]["service_level"]

        for product_index in range(self.n_products):
            self.service_level[product_index][0] = service_levels[product_index]
            self.service_level[product_index][1] = service_levels[product_index] 

            for tau_period in range(2, len(self.tau_periods)+1):
                self.service_level[product_index][tau_period] = self.service_level[product_index][tau_period-1] 

    def normalize_demand(self, products):
        products_reshaped = []
        for product in products:
            # Initialize the scaler
            scaler = StandardScaler()

            # Fit the scaler on your data
            scaler.fit(product.values.reshape(-1, 1))

            # Now you can use this scaler to transform your data
            normalized_sales_quantity = scaler.transform(product.values.reshape(-1, 1))

            # Convert the normalized numpy array back to Series
            normalized_series = pd.Series(normalized_sales_quantity.flatten(), index=product.index)
            # Add the normalized series to the list
            products_reshaped.append(normalized_series)
        return products_reshaped

    def set_demand_forecast(self, demand_forecast):
        self.demand_forecast = demand_forecast
        self.model.update()
        #print("demand forecast:")
        #print(self.demand_forecast)

    def set_holding_costs(self, unit_cost):
        # Multiply unit cost by 0.1 to get holding costs
        self.holding_cost = [0.1 * x for x in unit_cost]
        for product_index in range(self.n_products):
            self.shortage_cost[product_index] = self.holding_cost[product_index]/(1/self.service_level[product_index][self.tau_periods[0]] - 1)

    def set_safety_stock(self, standard_deviations):
        for product_index in range(self.n_products):
            for time_period in range(1, self.n_time_periods+1):
                self.safety_stock[product_index][time_period] = {}
                for tau_period in range(1, self.n_time_periods-time_period+2):            
                    squared_sum = sum(standard_deviations[product_index][time_period+t]**2 for t in range(0, tau_period))
                    self.safety_stock[product_index][time_period][tau_period] = norm.ppf(self.service_level[product_index][tau_period]) * np.sqrt(squared_sum) 
                    self.safety_stock[product_index][time_period][tau_period] = self.safety_stock[product_index][time_period][tau_period] / 10
        self.model.update()
        #print("safety stock")
        #print(self.safety_stock)
        
       



    def set_big_m(self):
        for product_index in (self.products): 
            for time_period in (self.time_periods):
                self.big_m[product_index][time_period] = {}
                for tau_period in (self.tau_periods[:self.n_time_periods - time_period+1]):
                    self.big_m[product_index][time_period][tau_period] = 0
                    if time_period == 0:
                        self.big_m[product_index][time_period][tau_period] = 0
                    else:
                        self.big_m[product_index][time_period][tau_period] = sum(self.demand_forecast[self.products[product_index]][self.time_periods[time_period+t]] for t in range(0, tau_period)) + self.safety_stock[product_index][time_period][tau_period]

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
        #minor_setup_incur = self.model.addConstrs((replenishment_q[product, time_period] <= self.big_m[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods), name="MinorSetupIncur")
        
        minor_setup_incur = self.model.addConstrs((replenishment_q[product, time_period] <= gp.quicksum((self.big_m[product][time_period][tau_period] * order_product[product, time_period, tau_period]  #blir det -1 her mtp index?
            for tau_period in self.tau_periods[:len(self.tau_periods) - time_period+1])) for product in self.products for time_period in self.time_periods), name="MinorSetupIncur")
        
        major_setup_incur = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for product in self.products for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= place_order[time_period] * self.n_products for time_period in self.time_periods), name="MajorSetupIncur")
        max_one_order = self.model.addConstrs((gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) <= 1 for product in self.products for time_period in self.time_periods), name="MaxOneOrder")
        
        if self.should_include_safety_stock:            
            minimum_inventory_ordering = self.model.addConstrs((inventory_level[self.products[p], self.time_periods[i]] >= 
                                                       order_product[self.products[p], self.time_periods[i], self.tau_periods[0]] * self.safety_stock[p][i][1]
                                                        
                                                   +gp.quicksum(order_product[self.products[p], self.time_periods[i], self.tau_periods[j-1]] * (self.safety_stock[p][i][j] 
                                                    + gp.quicksum(self.demand_forecast[self.products[p]][self.time_periods[i + t]] for t in range(1, j)))
                                                    for j in range(2, len(self.time_periods)-i+1))
                                                     for p in range(self.n_products) for i in range(1, self.n_time_periods+1)), name="minimumInventoryOrdering")
               
            minimum_inventory_not_ordering = self.model.addConstrs((inventory_level[self.products[p], self.time_periods[i]] >= self.safety_stock[p][i][1]
                                                                    for p in range (self.n_products) for i in range(1, self.n_time_periods+1)), name="minimumInventoryNotOrdering")

        else:
            minimum_inventory = self.model.addConstrs((inventory_level[product, time_period] >= 0 for product in self.products for time_period in self.time_periods[1:]), name="minimumInventory")
            
        # objective function
        obj = gp.quicksum(self.major_setup_cost * place_order[time_period] for time_period in self.time_periods) + gp.quicksum(self.minor_setup_cost[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in self.tau_periods[:len(self.tau_periods) - time_period]) for product in self.products for time_period in self.time_periods) + gp.quicksum(
            self.holding_cost[product] * inventory_level[product, time_period] for product in self.products for time_period in self.time_periods)

        self.model.setObjective(obj, GRB.MINIMIZE)
        self.has_been_set_up = True
