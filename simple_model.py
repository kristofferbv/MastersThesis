import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

#alt skal lages som en klasse med inputparametre: 
 # antall produkter
 # antall tidsperioder
 # majorsetup
 # minor setups
 # demand forecasts
 # holding costs
 # safety stock / parametre for å regne ut dette
 # bigM / dette kan sikkert regnes ut selv
 # inventory levels of period 0

rnd = np.random

n_time_periods = 2  # number of time periods
n_products = 2

# Sets
products = [i for i in range(1, n_products+1)]
time_periods = [i for i in range(0,n_time_periods + 1)]
tau_periods = [i for i in range(1, n_time_periods + 1)]

# Parameters
major_setup_cost = 100
minor_setup_cost = {i: rnd.randint(1,10) for i in products}
demand_forecast = {(i,j): rnd.randint(1,10) for i in products for j in time_periods}
holding_cost = {i: rnd.random() for i in products}
safety_stock = {(i,j,k): rnd.random() for i in products for j in time_periods for k in tau_periods}
bigM = {i: rnd.random()*10 for i in products}


# create model
inventoryModel = gp.Model('Inventory Control 1')

#create variables
replenishment_q = inventoryModel.addVars(products, time_periods, lb= 0, name="ReplenishmentQ")
order_product = inventoryModel.addVars(products, time_periods, tau_periods, vtype=GRB.BINARY, name="OrderProduct")
place_order = inventoryModel.addVars(time_periods, vtype=GRB.BINARY, name="PlaceOrder")
inventory_level = inventoryModel.addVars(products, time_periods, lb=safety_stock, name="InventoryLevel")

#constraints
# start inventory constraint
# this will start as a parameter 
startInventory = inventoryModel.addConstrs((inventory_level[product, time_periods[0]] == 3) for product in products)

# må ha med tidsperiode 0 og at denne gjelder fra 1 for å ikke å feil
inventory_balance = inventoryModel.addConstrs((inventory_level[product, time_periods[i-1]] + replenishment_q[product, time_periods[i]] == demand_forecast[(product, time_periods[i])] + inventory_level[product, time_periods[i]] for product in products for i in range(1,len(time_periods))), name="InventoryBalance")
# skal ikke ha tau i time periods, må finne ut hvordan man gjør sum fra til
minor_setup_incur = inventoryModel.addConstrs((replenishment_q[product,time_period] <= bigM[product] * gp.quicksum(order_product[product,time_period,tau_period] for tau_period in tau_periods[:len(time_periods)-time_period]) for product in products for time_period in time_periods),  name="MinorSetupIncur")

# skal ikke ha tau i time periods, må finne ut hvordan man gjør sum fra til
major_setup_incur = inventoryModel.addConstrs((gp.quicksum(order_product[product,time_period,tau_period] for product in products for tau_period in tau_periods[:len(time_periods)-time_period]) <= place_order[time_period] * n_products for time_period in time_periods), name= "MajorSetupIncur")

# enten =1 og ta fra tau=0 eller <=1 og ta fra tau = 1
max_one_order = inventoryModel.addConstrs((gp.quicksum(order_product[product,time_period,tau_period] for tau_period in tau_periods[:len(time_periods)-time_period]) == 1 for product in products  for time_period in time_periods), name= "MaxOneOrder")

#summene blir feil her, må finne eks på sum fra 1 til .. 
minimum_inventory = inventoryModel.addConstrs((inventory_level[product, time_period] >= (1- gp.quicksum(order_product[product, time_period, tau_period] for tau_period in tau_periods[:len(time_periods)-time_period])) * safety_stock[product, time_period, 1] + gp.quicksum(order_product[product, time_period, tau_period]*(safety_stock[product, time_period, tau_period] + gp.quicksum(demand_forecast[(product, time_period + x)] for x in range(1,tau_period))) for tau_period in tau_periods[:len(time_periods)-time_period]) for product in products for time_period in time_periods), name="minimumInventory")

#objective function
obj = gp.quicksum(major_setup_cost*place_order[time_period] for time_period in time_periods) + gp.quicksum(minor_setup_cost[product] * gp.quicksum(order_product[product, time_period, tau_period] for tau_period in tau_periods[:len(time_periods)-time_period]) for product in products for time_period in time_periods) + gp.quicksum(holding_cost[product]*inventory_level[product, time_period] for product in products for time_period in time_periods)


inventoryModel.setObjective(obj, GRB.MINIMIZE)
inventoryModel.optimize()


