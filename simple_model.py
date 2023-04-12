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

 # hente dette fra en fil?



import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

# alt skal lages som en klasse med inputparametre:
# antall produkter
# antall tidsperioder
# tau-parameter
# majorsetup
# minor setups
# demand forecasts
# holding costs
# safety stock / parametre for å regne ut dette
# bigM / dette kan sikkert regnes ut selv
# inventory levels of period 0

rnd = np.random

n_time_periods = 2  # number of time periods
n_products = 3

# Sets
products = [i for i in range(1, n_products + 1)]
time_periods = [i for i in range(0, n_time_periods + 1)]
tau_periods = [i for i in range(1, n_time_periods + 1)]

# Parameters
#major_setup_cost = 100
#minor_setup_cost = {i: rnd.randint(1, 10) for i in products}

#demand_forecast = {(i, j): rnd.randint(1, 10) for i in products for j in time_periods}
#print(demand_forecast)
 
#holding_cost = {i: rnd.random() for i in products}
#safety_stock = {(i, j, k): rnd.random() for i in products for j in time_periods for k in tau_periods}
#bigM = {i: rnd.random() * 10 for i in products}

major_setup_cost = 100

minor_setup_cost = {1:10, 2:20, 3:30}

demand_forecast = {(1,1):10 , (1,2):10, (2,1):20, (2,2):20, (3,1):25, (3,2):10}

holding_cost = {1: 2, 2:3, 3:1}

safety_stock = {(1,1,1): 0, (1,1,2):0, (1,2,1):0, (2,1,1): 0, (2,1,2):0, (2,2,1):0, (3,1,1): 0, (3,1,2):0, (3,2,1):0 }

bigM = {1: 100, 2:100, 3: 100}

# create model
inventoryModel = gp.Model('Inventory Control 1')

# create variables
replenishment_q = inventoryModel.addVars(products, time_periods[1:], lb=0, name="ReplenishmentQ")
order_product = inventoryModel.addVars(products, time_periods[1:], tau_periods, vtype=GRB.BINARY, name="OrderProduct")
place_order = inventoryModel.addVars(time_periods[1:], vtype=GRB.BINARY, name="PlaceOrder")
inventory_level = inventoryModel.addVars(products, time_periods, lb=safety_stock, name="InventoryLevel")


# constraints
# start inventory constraint
# this will start as a parameter
startInventory = inventoryModel.addConstrs((inventory_level[product, time_periods[0]] == 0) for product in products)

inventory_balance = inventoryModel.addConstrs((inventory_level[product, time_periods[i - 1]] + replenishment_q[product, time_periods[i]] == demand_forecast[(product, time_periods[i])] + inventory_level[product, time_periods[i]] for product in products for i in range(1, len(time_periods))), name="InventoryBalance")
minor_setup_incur = inventoryModel.addConstrs((replenishment_q[product, time_periods[i]] <= bigM[product] * gp.quicksum(order_product[product, time_periods[i], tau_period] for tau_period in tau_periods[:(len(tau_periods) - time_periods[i])]) for product in products for i in range(1, len(time_periods))), name="MinorSetupIncur")
major_setup_incur = inventoryModel.addConstrs((gp.quicksum(order_product[product, time_periods[i], tau_period] for product in products for tau_period in tau_periods[:len(tau_periods) - time_periods[i]]) <= place_order[time_periods[i]] * n_products for i in range(1, len(time_periods))), name="MajorSetupIncur")
max_one_order = inventoryModel.addConstrs((gp.quicksum(order_product[product, time_periods[i], tau_period] for tau_period in tau_periods[:len(tau_periods) - time_periods[i]]) <= 1 for product in products for i in range(1, len(time_periods))), name="MaxOneOrder")
minimum_inventory = inventoryModel.addConstrs(
    (inventory_level[product, time_periods[i]] >= (1 - gp.quicksum(order_product[product, time_periods[i], tau_period] for tau_period in tau_periods[:len(tau_periods)- time_periods[i]])) * safety_stock[product, time_periods[i], 1]
      + gp.quicksum(order_product[product, time_periods[i], tau_period] * 
    (safety_stock[product, time_periods[i], tau_period] + gp.quicksum(demand_forecast[(product, time_periods[i] + x)] for x in range(1, tau_period))) for tau_period in tau_periods[:len(time_periods) - time_periods[i]]) for product
    in products for i in range(1, len(time_periods))), name="minimumInventory")



# objective function
obj = gp.quicksum(major_setup_cost * place_order[time_periods[i]] for i in range(1, len(time_periods))) + gp.quicksum(minor_setup_cost[product] * gp.quicksum(order_product[product, time_periods[i], tau_period] for tau_period in tau_periods[:(len(tau_periods)-time_periods[i])]) for product in products for i in range(1, len(time_periods))) + gp.quicksum(holding_cost[product] * inventory_level[product, time_periods[i]] for product in products for i in range(1, len(time_periods)))

inventoryModel.setObjective(obj, GRB.MINIMIZE)
inventoryModel.optimize()

for v in inventoryModel.getVars():
    print(v.x)