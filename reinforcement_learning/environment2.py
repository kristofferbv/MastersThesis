import copy
import os
from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler
import config_utils
from MIP.forecasting import sarima, holt_winters_method
from generate_data import generate_next_week_demand
import statsmodels.api as sm


class JointReplenishmentEnv(gym.Env, ABC):
    def __init__(self, products):
        super(JointReplenishmentEnv, self).__init__()

        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # # construct the path to the file
        # file_path = os.path.join(current_dir, '../config.yml')
        # config = config_utils.load_config(file_path)
        config = config_utils.load_config("config.yml")
        rl_config = config["rl_model"]
        env_config = config["environment"]
        self.verbose = False
        self.products = products
        self.scaled_products = self.products #self.normalize_demand(products[:])
        # starting to learn from first period then moving on
        self.time_period = 208
        self.forecasted = False
        # Parameters
        self.major_setup_cost = rl_config["joint_setup_cost"]
        # self.minor_setup_cost = rl_config["minor_setup_cost"]
        self.minor_setup_ratio = rl_config["minor_setup_ratio"]
        self.minor_setup_cost = [self.minor_setup_ratio * self.major_setup_cost / len(self.products) for i in range(0, len(self.products))]

        self.safety_stock = {}
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.n_periods = rl_config["n_time_periods"]

        self.n_periods_historical_data = env_config["n_periods_historical_data"]
        self.rolling_window = env_config["rolling_window_forecast"]
        self.should_include_individual_forecast = env_config["should_include_individual_forecast"]  # Should include forecast for the specific product as part of the state
        self.should_include_total_forecast = env_config["should_include_total_forecast"]  # Should include total forecast for all products as part of the state

        # Define action and observation spaces
        self.n_action_classes = env_config["n_action_classes"]
        self.max_order_quantity = env_config["maximum_order_quantity"]
        self.action_multiplier = self.max_order_quantity / self.n_action_classes
        if not self.action_multiplier.is_integer():
            raise Exception("maximum_order_quantity / n_action_classes must be an integer")
        self.counter = 0
        self.forecast = []
        self.forecast2 = {}


        self.action_space = gym.spaces.Discrete(self.n_action_classes)  # 10 discrete actions from 0 to 9 inclusive
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=( self.n_periods_historical_data + self.should_include_individual_forecast + self.should_include_total_forecast, 2*len(products)), dtype=np.float32)
        self.forecast = {}
        self.inventory_levels = {}
        self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period

        self.inventory_levels[self.current_period] = [0 for _ in self.products]
        self.reset()

    def increase_time_period(self, increase):
        self.time_period += increase
    def reset_time_period(self):
        self.time_period = 0
    def set_costs(self, products, mult = 1):
        unit_costs = [df.iloc[0]['average_unit_price'] for df in products]
        if mult != 1:
            self.holding_cost = [0.1 * x for x in unit_costs]
            self.minor_setup_cost = [2.5 * self.major_setup_cost / len(self.products) for i in range(0, len(self.products))]

        else:
            self.holding_cost = [0.1 * x for x in unit_costs]
            self.minor_setup_cost = [self.minor_setup_ratio * self.major_setup_cost / len(self.products) for i in range(0, len(self.products))]


        # Calculate shortage costs
        self.shortage_cost = []
        for product_index in range(len(products)):
            self.shortage_cost.append(self.holding_cost[product_index] / (1 / 0.95 - 1))




    def reset(self, **kwargs):
        # Reset the environment to the initial state. Setting start period so that we can ensure we have all historical data required for the first state
        self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period
        # self.inventory_levels = [0 for _ in self.products]
        self.counter = 0
        return self._get_observation()

    def reset_inventory(self):
        self.inventory_levels[self.current_period] = [0 for _ in self.products]

    def step(self, action):
        # Need to discretize the actions.
        # action = [x * self.action_multiplier for x in action]
        # Apply the replenishment action
        major_setup_triggered = False

        # action = [round(action / 2) * 2 for action in action]

        individual_rewards = []
        count_major_setup_sharing = len([i for i in action if i > 0])
        inventory_levels = self.inventory_levels[self.current_period]
        demands = []
        minor_costs = []
        major_costs = []
        holding_costs = []
        shortage_costs = []
        rewards = []

        for i, product in enumerate(self.products):
            if action[i] > 0:
                action_value = 0
                # for i in range(1, action[i]):
                    # action_value += self.forecast2[product["product_hash"].iloc[1]][min(53,self.counter + i)]
                # print(f"forecast for {action[i]} periods: ", action_value)
                inventory_levels[i] += action[i]
                # Apply only fractional part of major setup costs corresponding to number of products ordering
                # if count_major_setup_sharing == 1:
                #     major_cost = self.major_setup_cost / count_major_setup_sharing * 4
                # else:
                major_cost = self.major_setup_cost / count_major_setup_sharing

                minor_cost = self.minor_setup_cost[i]
            else:
                major_cost = 0
                minor_cost = 0

            # Simulate demand and calculate shortage cost and holding cost.
            try:
                demand = product.iloc[self.current_period]["sales_quantity"]
                # print(demand)

            except:
                print(self.current_period)
            shortage_cost = abs(min((inventory_levels[i] - demand), 0)) * self.shortage_cost[i]
            inventory_levels[i] = max(inventory_levels[i] - demand, 0)
            holding_cost = inventory_levels[i] * self.holding_cost[i]
            # Calculate the total cost
            total_cost = minor_cost + major_cost + shortage_cost + holding_cost
            if self.verbose:
                shortage_costs.append(shortage_cost)
                minor_costs.append(minor_cost)
                major_costs.append(major_cost)
                holding_costs.append(holding_cost)
                demands.append(demand)
                rewards.append(total_cost)

            # Calculate individual reward for this product
            individual_rewards.append(-total_cost)
        if self.verbose:
            print(f"Demand {self.current_period} {demands}")
            print(f"shortage {shortage_costs}")
            print(f" minor: {minor_costs}")
            print(f"major: {major_costs}")
            print(f"holding{holding_costs}")
            print(f"costs: {rewards}")
            print(f"TOTAL: {sum(individual_rewards)}")

        # Update the current period
        self.current_period += 1
        self.inventory_levels[self.current_period] = inventory_levels
        done = self.current_period == self.n_periods + max(self.rolling_window, self.n_periods_historical_data) + self.time_period
        self.inventory_levels[self.current_period] = inventory_levels
        return self._get_observation(), individual_rewards, done, {}

    def _get_observation(self):
        # Create an observation of the stock levels for the last n_periods_lookahead and the current inventory levels
        observation = []
        start_date = self.products[0].index[self.current_period]
        has_counted = False

        for i, product in enumerate(self.scaled_products):
            if not self.forecasted:
                self.forecast2[product["product_hash"].iloc[1]], _ = holt_winters_method.forecast(self.products[i], start_date, n_time_periods=53)

            historical_demand = product["sales_quantity"].iloc[max(self.current_period - self.n_periods_historical_data+1, 0):self.current_period + 1].values

            if len(historical_demand) < self.n_periods_historical_data:
                historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)

            # Prepare an array for this product with dimensions (13, 2)
            product_array = np.zeros((13, 2))

            # Fill in the historical demand values in the first column
            product_array[:, 0] = historical_demand # or fill with zeros if less than 13 data points are available

            # Fill in the inventory levels in the second column
            inventory_levels = [self.inventory_levels[j][i] for j in range(max(self.current_period - 12, self.time_period + self.n_periods_historical_data), self.current_period + 1)]
            if len(inventory_levels) < 13:
                inventory_levels = [0] * (13 - len(inventory_levels)) + inventory_levels
            product_array[:, 1] = inventory_levels

            observation.append(product_array)


        return observation

