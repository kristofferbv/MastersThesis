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
        rl_config = config["rl_model"]

        # Parameters
        self.major_setup_cost = rl_config["joint_setup_cost"]
        self.minor_setup_cost = rl_config["minor_setup_cost"]
        self.holding_cost = rl_config["holding_cost"]
        self.shortage_cost = rl_config["holding_cost"]
        self.safety_stock = {}
        self.big_m = rl_config["big_m"]
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.n_periods = rl_config["n_time_periods"]
        self.n_periods_historical_data = config["environment"]["n_periods_historical_data"]
        self.products = products

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(10)  # 10 discrete actions from 0 to 9 inclusive
        # action_dimensions = [len(product) for product in products]
        # print(action_dimensions)
        # self.action_space = spaces.Tuple([spaces.Discrete(dim) for dim in action_dimensions])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(products), self.n_periods_historical_data + 3), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment to the initial state. Setting start period so that we can ensure we have all historical data required for the first state
        self.current_period = self.n_periods_historical_data
        self.inventory_levels = [0 for _ in self.products]

        return self._get_observation()

    def step(self, action):
        # Apply the replenishment action
        major_setup_triggered = False
        individual_rewards = []
        count_major_setup_sharing = len([i for i in action if i > 0])

        for i, product in enumerate(self.products):
            if action[i] > 0:
                self.inventory_levels[i] += action[i]
                # Apply only fractional part of major setup costs corresponding to number of products ordering
                major_cost = self.major_setup_cost / count_major_setup_sharing
                minor_cost = self.minor_setup_cost[i]
            else:
                major_cost = 0
                minor_cost = 0

            # Simulate demand and calculate shortage cost and holding cost.
            demand = product.iloc[self.current_period]['sales_quantity'] / 10  # dividing by 10 for training purpose only
            shortage_cost = abs(min((self.inventory_levels[i] - demand), 0)) * self.shortage_cost[i]
            self.inventory_levels[i] = max(self.inventory_levels[i] - demand, 0)
            holding_cost = self.inventory_levels[i] * self.holding_cost[i]

            # Calculate the total cost
            total_cost = minor_cost + major_cost + shortage_cost + holding_cost

            # Calculate individual reward for this product
            individual_rewards.append(-total_cost)

        # Update the current period
        self.current_period += 1
        done = self.current_period == self.n_periods + self.n_periods_historical_data

        return self._get_observation(), individual_rewards, done, {}

    def calculate_reward(self, action):
        major_setup_triggered = any(action)
        individual_order_penalty = 0
        minor_cost = 0
        for i, product in enumerate(self.products):
            if action[i] > 0:
                minor_cost += self.minor_setup_cost[i]
                if not major_setup_triggered:
                    major_setup_triggered = True
                else:
                    individual_order_penalty += self.individual_order_penalty[i]
        ...
        total_cost = minor_cost + major_cost + shortage_cost + holding_cost + individual_order_penalty
        return -total_cost

    def _get_observation(self):
        # Create an observation of the stock levels for the last n_periods_lookahead and the current inventory levels
        observation = []
        total_forecast = 0
        for i, product in enumerate(self.products):
            historical_demand = product.iloc[max(self.current_period - self.n_periods_historical_data, 0):self.current_period]['sales_quantity'].values / 10
            forecast = sum(historical_demand) / len(historical_demand)
            total_forecast += forecast
            if len(historical_demand) < self.n_periods_historical_data:
                historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            inv = [self.inventory_levels[i], forecast]
            observation.append(np.concatenate((inv, historical_demand)))

            # Pad with zeros if there are not enough historical periods
            # if len(historical_demand) < self.n_periods_historical_data:
            #     historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            # observation.append([self.inventory_levels[i], forecast])
        # appending total forecast as part of the state in the hope of coordinating the agents better

        # observation = [x + [total_forecast] for x in observation]
        observation = [np.append(arr, total_forecast) for arr in observation]

        # observation = np.array([np.append(arr, total_forecast) for arr in observation])

        return np.array(observation)
