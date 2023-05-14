import os
from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.preprocessing import StandardScaler
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
        self.shortage_cost = rl_config["shortage_cost"]
        self.safety_stock = {}
        self.big_m = rl_config["big_m"]
        self.start_inventory = [0, 0, 0, 0, 0, 0]
        self.n_periods = rl_config["n_time_periods"]
        self.n_periods_historical_data = config["environment"]["n_periods_historical_data"]
        self.rolling_window = config["environment"]["rolling_window_forecast"]

        self.products = products
        self.scaled_products = self.normalize_demand(products[:])

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(10)
        # action_dimensions = [len(product) for product in products]
        # print(action_dimensions)
        # self.action_space = spaces.Tuple([spaces.Discrete(dim) for dim in action_dimensions])
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(len(products), self.n_periods_historical_data + 3), dtype=np.float32)

        self.reset()

    def reset(self, **kwargs):
        # Reset the environment to the initial state. Setting start period so that we can ensure we have all historical data required for the first state
        self.current_period = max(self.rolling_window, self.n_periods_historical_data)
        self.inventory_levels = [0 for _ in self.products]

        return self._get_observation()

    def step(self, action):
        epoch = action.pop()
        # Apply the replenishment action
        major_setup_triggered = False
        for i, product in enumerate(self.products):
            if action[i] > 0:
                self.inventory_levels[i] += action[i]
                major_setup_triggered = True
        # Calculate the minor and major setup costs
        minor_cost = np.sum((a >= 1) * b for a, b in zip(action, self.minor_setup_cost))
        major_cost = self.major_setup_cost if major_setup_triggered else 0

        # Simulate demand and calculate shortage cost and holding cost.
        shortage_cost = 0
        holding_cost = 0
        if epoch == 100:
            print("tidssteg: ", self.current_period - max(self.rolling_window, self.n_periods_historical_data))
            print("inventory level ", self.inventory_levels)
        for i, product in enumerate(self.products):
            demand = product.iloc[self.current_period] / 10  # dividing by 10 for training purpose only
            if epoch == 100:
                print("demand product " + str(i) + ":", demand)
                print("shortage", self.inventory_levels[i] - demand)
            shortage_cost += abs(min((self.inventory_levels[i] - demand), 0)) * self.shortage_cost[i]
            self.inventory_levels[i] = max(self.inventory_levels[i] - demand, 0)
            holding_cost += self.inventory_levels[i] * self.holding_cost[i]

            # Calculate the total cost
        total_cost = minor_cost + major_cost + shortage_cost + holding_cost

        # Update the current period
        self.current_period += 1
        done = self.current_period == self.n_periods + max(self.rolling_window, self.n_periods_historical_data)
        if epoch == 100:
            print("inventory after demand and action", self.inventory_levels)
            print("actions", action)
            print("total costs", total_cost)

        return self._get_observation(), -total_cost, done, {}

    def _get_observation(self):
        # Create an observation of the stock levels for the last n_periods_lookahead and the current inventory levels
        observation = []
        total_forecast = 0
        for i, product in enumerate(self.scaled_products):
            historical_demand = product.iloc[max(self.current_period - self.n_periods_historical_data, 0):self.current_period].values / 10
            forecast_demand = product.iloc[max(self.current_period - self.rolling_window, 0):self.current_period].values / 10
            forecast = sum(forecast_demand) / len(forecast_demand)
            total_forecast += forecast
            if len(historical_demand) < self.n_periods_historical_data:
                historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            inv = [self.inventory_levels[i] / 10, forecast]
            observation.append(np.concatenate((inv, historical_demand)))
            # observation.append(np.concatenate(([self.inventory_levels[i]], historical_demand)))

            # Pad with zeros if there are not enough historical periods
            # if len(historical_demand) < self.n_periods_historical_data:
            #     historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            # observation.append([self.inventory_levels[i], forecast])
        # appending total forecast as part of the state in the hope of coordinating the agents better

        # observation = [x + [total_forecast] for x in observation]
        observation = [np.append(arr, total_forecast) for arr in observation]

        # observation = np.array([np.append(arr, total_forecast) for arr in observation])

        return np.array(observation)

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
