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


#sets and parameters
products = ["p1","p2"]

# need to define this such that it is possible to get the invnetory balancing constraints
# perhaps better with numbers
timePeriods = ["t0", "t1","t2"]

nTimePeriods = 2

tauPeriods = ["tau0","tau1","tau2"]

majorSetup = 100

minorSetup = {"p1": 10, "p2":20}

demandForecast = {("p1","t1"):3, ("p1", "t2"):5, 
                  ("p2","t1"):1, ("p2", "t2"):2,}

holdingCost = {"p1": 2, "p2":3}

safetyStock = {"p1": 3, "p2":2}

bigM = {"p1": 15, "p2":5}


# create model
inventoryModel = gp.Model('Inventory Control 1')

#create variables
replenishmentQ = inventoryModel.addVars(products, timePeriods, lb= 0, name="ReplenishmentQ")

orderProduct = inventoryModel.addVars(products, timePeriods, tauPeriods, vtype=GRB.BINARY, name="OrderProduct")

placeOrder = inventoryModel.addVars(timePeriods, vtype=GRB.BINARY, name="PlaceOrder")

inventoryLevel = inventoryModel.addVars(products, timePeriods, lb=safetyStock, name="InventoryLevel")


#constraints

# start inventory constraint
# this will start as a parameter 
startInventory = inventoryModel.addConstrs((inventoryLevel[product, "t0"] == 3) for product in products)


# må ha med tidsperiode 0 og at denne gjelder fra 1 for å ikke å feil
inventoryBalance = inventoryModel.addConstrs((inventoryLevel[product, timePeriod[-1]] + replenishmentQ[product, timePeriod] == demandForecast[product, timePeriod] + inventoryLevel[product, timePeriod] for product in products for timePeriod in timePeriods if timePeriod != timePeriod[0]), name="InventoryBalance")

# skal ikke ha tau i time periods, må finne ut hvordan man gjør sum fra til
minorSetupIncur = inventoryModel.addConstrs((replenishmentQ[product,timePeriod] <= bigM(product) * gp.quicksum(orderProduct[product][timePeriod][tauPeriod] for tauPeriod in timePeriods) for product in products for timePeriod in timePeriods),  name="MinorSetupIncur")

# skal ikke ha tau i time periods, må finne ut hvordan man gjør sum fra til
majorSetupIncur = inventoryModel.addConstrs((gp.quicksum(orderProduct[product][timePeriod][tau] for product in products for tau in timePeriods) <= placeOrder[timePeriod] for timePeriod in timePeriods), name= "MajorSetupIncur")


# enten =1 og ta fra tau=0 eller <=1 og ta fra tau = 1
maxOneOrder = inventoryModel.addConstrs((gp.quicksum(orderProduct[product][timePeriod][tau] for tau in timePeriods) == 1  for product in products  for timePeriod in timePeriods), name= "MaxOneOrder")


#summene blir feil her, må finne eks på sum fra 1 til .. 
minimumInventory = inventoryModel.addConstrs((inventoryLevel[product, timePeriod] >= orderProduct[product, timePeriod, "tau0"] * safetyStock[product, timePeriod, "tau1"] + gp.quicksum(orderProduct[product, timePeriod, tauPeriod]*(safetyStock[product, timePeriod, tauPeriod] + gp.quicksum(demandForecast[product, timePeriod, 1+x] for x in range(1,tauPeriod) for tauPeriod in tauPeriods) )) for product in products for timePeriod in timePeriods), name="minimumInventory")


#objective function

obj = gp.quicksum(majorSetup*placeOrder[timePeriod] for timePeriod in timePeriods) + gp.quicksum(minorSetup * gp.quicksum(orderProduct[product, timePeriod, tauPeriod] for tauPeriod in tauPeriods) for product in products for timePeriod in timePeriods) + gp.quicksum(holdingCost[product]*inventoryLevel[product, timePeriod] for product in products for timePeriod in timePeriods)


inventoryModel.setObjective(obj, GRB.MINIMIZE)

inventoryModel.optimize()


