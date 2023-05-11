import os
from abc import ABC

import gym
import numpy as np
from gym import spaces

import config_utils


class JointReplenishmentEnv(gym.Env, ABC):
    def __init__(self, products):
        super(JointReplenishmentEnv, self).__init__()

        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # # construct the path to the file
        # file_path = os.path.join(current_dir, '../config.yml')
        # config = config_utils.load_config(file_path)
        config = config_utils.load_config("config.yml")

        # Parameters
        self.major_setup_cost = config["deterministic_model"]["joint_setup_cost"]
        self.minor_setup_cost = config["deterministic_model"]["minor_setup_cost"]
        self.holding_cost = config["deterministic_model"]["holding_cost"]
        self.shortage_cost = config["deterministic_model"]["holding_cost"]
        self.safety_stock = {}
        self.big_m = config["deterministic_model"]["big_m"]
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.n_periods = config["deterministic_model"]["n_time_periods"]
        self.n_periods_historical_data = config["environment"]["n_periods_historical_data"]
        self.products = products

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([100 for i in range(len(products))])
        # action_dimensions = [len(product) for product in products]
        # print(action_dimensions)
        # self.action_space = spaces.Tuple([spaces.Discrete(dim) for dim in action_dimensions])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(products), self.n_periods_historical_data + 1), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment to the initial state
        self.current_period = 0
        self.inventory_levels = [0 for _ in self.products]

        return self._get_observation()

    def step(self, action):
        # Apply the replenishment action
        major_setup_triggered = False
        for i, product in enumerate(self.products):
            if action[i] > 0:
                self.inventory_levels[i] += action[i]
                major_setup_triggered = True

        # Calculate the minor and major setup costs
        minor_cost = np.sum(a*b for a, b in zip(action, self.minor_setup_cost))
        major_cost = self.major_setup_cost if major_setup_triggered else 0

        # Simulate demand and calculate shortage cost and holding cost.
        shortage_cost = 0
        holding_cost = 0
        for i, product in enumerate(self.products):
            demand = product.iloc[self.current_period]['sales_quantity']
            self.inventory_levels[i] = max(self.inventory_levels[i] - demand, 0)
            shortage_cost += (demand - self.inventory_levels[i]) * self.shortage_cost[i]
            holding_cost += self.inventory_levels[i] * self.holding_cost[i]

        # Update the current period
        self.current_period += 1
        done = self.current_period > self.n_periods
        # done = self.current_period >= len(self.products[0])

        # Calculate the total cost
        total_cost = minor_cost + major_cost + shortage_cost

        return self._get_observation(), -total_cost, done, {}

    def _get_observation(self):
        # Create an observation of the stock levels for the last n_periods_lookahead and the current inventory levels
        observation = []
        for i, product in enumerate(self.products):
            historical_demand = product.iloc[max(self.current_period - self.n_periods_historical_data, 0):self.current_period]['sales_quantity'].values
            # Pad with zeros if there are not enough historical periods
            if len(historical_demand) < self.n_periods_historical_data:
                historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            observation.append(np.concatenate(([self.inventory_levels[i]], historical_demand)))
        return np.array(observation)
