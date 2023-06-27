import copy
from abc import ABC

import gym
import numpy as np
import pandas as pd
import statsmodels.api as sm
from gym import spaces
from sklearn.preprocessing import StandardScaler

import config_utils
from MIP.forecasting_methods import holt_winters_method


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
        self.should_reset_time_at_each_episode = env_config["should_reset_at_each_episode"]
        self.verbose = False
        self.products = products
        self.scaled_products = self.products  # self.normalize_demand(products[:])
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
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.n_periods_historical_data + 1 + self.should_include_individual_forecast + self.should_include_total_forecast, len(products)), dtype=np.float32)
        self.forecast = {}
        self.inventory_levels = [0 for _ in self.products]
        self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period
        self.steps = 0
        self.reset()
        self.fulfilled_demand = {}
        self.achieved_service_level = {}

        self.real_holding_costs = 0
        self.real_shortage_costs = 0
        self.real_setup_costs = 0

    def get_costs(self):
        return self.real_holding_costs, self.real_shortage_cots, self.real_setup_costs
    def reset_current_period(self):
        self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period
    def reset_costs(self):
        self.real_holding_costs = 0
        self.real_shortage_costs = 0
        self.real_setup_costs = 0


    def increase_time_period(self, increase):
        self.time_period += increase

    def reset_time_period(self):
        self.time_period = 0

    def set_costs(self, products, mult=1):
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
            print(self.holding_cost[product_index])
            self.shortage_cost.append(self.holding_cost[0]/ (1 / 0.95 - 1))


    def reset(self, **kwargs):
        # Reset the environment to the initial state. Setting start period so that we can ensure we have all historical data required for the first state
        # self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period
        # self.inventory_levels = [0 for _ in self.products]
        if self.should_reset_time_at_each_episode:
            self.current_period = max(self.rolling_window, self.n_periods_historical_data) + self.time_period

        # self.counter = 0
        self.steps = 0
        return self._get_observation()

    def reset_inventory(self):
        self.inventory_levels = [0 for _ in self.products]

    def step(self, action):
        self.steps += 1
        individual_rewards = []
        count_major_setup_sharing = len([i for i in action if i > 0])

        demands = []
        minor_costs = []
        major_costs = []
        holding_costs = []
        shortage_costs = []
        rewards = []
        self.fulfilled_demand = {}
        self.achieved_service_level = {}
        # self.inventory_levels = max([0,0], [self.inventory_levels[0]-5, self.inventory_levels[1]-5])




        for i, product in enumerate(self.products):
            # self.verbose = True
            if action[i] > 0:
                self.inventory_levels[i] += action[i]
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
            shortage_cost = abs(min((self.inventory_levels[i] - demand), 0)) * self.shortage_cost[i]
            self.fulfilled_demand[i] = min(self.inventory_levels[i], demand)
            if demand != 0:
                self.achieved_service_level[i] = self.fulfilled_demand[i] / demand
            else:
                self.achieved_service_level[i] = 1

            self.inventory_levels[i] = max(self.inventory_levels[i] - demand, 0)

            holding_cost = self.inventory_levels[i] * self.holding_cost[i]
            # Calculate the total cost
            total_cost = minor_cost + major_cost + shortage_cost + holding_cost
            # if self.achieved_service_level[i]<0.5:
            # total_cost += min(0,(0.9 -self.achieved_service_level[i])) * 5000
            # if self.inventory_levels[i] < 5 and self.steps > 1:
            #     total_cost += 100

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
            print(f"Inventory levels: {self.inventory_levels}")
            print(f"shortage {shortage_costs}")
            print(f" minor: {minor_costs}")
            print(f"major: {major_costs}")
            print(f"holding{holding_costs}")
            print(f"costs: {rewards}")
            print(f"TOTAL: {sum(individual_rewards)}")
        self.real_setup_costs += sum(minor_costs + major_costs)
        self.real_shortage_costs += sum(shortage_costs)
        self.real_holding_costs += sum(holding_costs)
        # Update the current period

        self.current_period += 1
        done = self.steps == 51 #self.current_period == self.n_periods + max(self.rolling_window, self.n_periods_historical_data) + self.time_period
        return self._get_observation(), individual_rewards, done, {}

    def _get_observation(self):
        # Create an observation of the stock levels for the last n_periods_lookahead and the current inventory levels
        observation = []
        inventory = []
        demand = []
        total_forecast = 0
        forecast = 0
        start_date = self.products[0].index[self.current_period]
        has_counted = False
        for i, product in enumerate(self.scaled_products):
            if not self.forecasted:
                self.forecast2[product["product_hash"].iloc[1]], _ = holt_winters_method.forecast(self.products[i], start_date, n_time_periods=53)

            historical_demand = product["sales_quantity"].iloc[max(self.current_period - self.n_periods_historical_data, 0):self.current_period].values
            if self.should_include_individual_forecast or self.should_include_total_forecast:
                if product["product_hash"].iloc[1] not in self.forecast.keys():
                    print(product["product_hash"].iloc[1])
                    self.forecast[product["product_hash"].iloc[1]], _ = holt_winters_method.forecast(product, start_date, n_time_periods=53)
                forecast = []
                forecast = self.forecast[product["product_hash"].iloc[1]][self.counter]
                if not has_counted:
                    self.counter += 1
                    has_counted = True
                # forecast_demand = product["sales_quantity"].iloc[max(self.current_period - self.rolling_window, 0):self.current_period].values
                # forecast = sum(forecast_demand) / len(forecast_demand)
                total_forecast += self.forecast[product["product_hash"].iloc[1]][self.counter]
            if len(historical_demand) < self.n_periods_historical_data:
                historical_demand = np.pad(historical_demand, (self.n_periods_historical_data - len(historical_demand), 0), mode='constant', constant_values=0)
            # Divide inventory level by 10 in an attempt to normalize the data. Might not be helpful
            if self.should_include_individual_forecast:
                # concat = np.concatenate((np.array([self.inventory_levels[i] / 10]), forecast)).tolist()
                concat = [self.inventory_levels[i] / 10, forecast]
                observation.append(np.concatenate((concat, historical_demand)))
            else:
                inventory.append(self.inventory_levels[i])
                demand.append(historical_demand)
                observation.append(np.concatenate((historical_demand, [self.inventory_levels[i]])))
        if self.should_include_total_forecast:
            observation = [np.append(arr, total_forecast) for arr in observation]
        self.forecasted = True

        return np.array(observation)  # (inventory, demand)

    def normalize_demand(self, products):

        products_copy = copy.deepcopy(products)
        products_reshaped = []
        for product in products_copy:
            # Initialize the scaler
            self.scaler = StandardScaler()
            res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=52)
            seasonal_filled = res.seasonal.fillna(method='ffill')

            product["sales_quantity_diff"] = product["sales_quantity"].diff()
            product["sales_quantity"] = seasonal_filled

            # Fit the scaler on your data
            self.scaler.fit(product["sales_quantity"].values.reshape(-1, 1))

            # Now you can use this scaler to transform your data
            normalized_sales_quantity = self.scaler.transform(product["sales_quantity"].values.reshape(-1, 1))

            # Convert the normalized numpy array back to Series
            normalized_series = pd.Series(normalized_sales_quantity.flatten(), index=product.index)
            # Add the normalized series to the list
            product["sales_quantity"] = normalized_series
            products_reshaped.append(product)
        return products_reshaped

    def scale_demand(self, products):
        products_copy = copy.deepcopy(products)
        products_reshaped = []
        for product in products_copy:
            # Now you can use this scaler to transform your data
            res = sm.tsa.seasonal_decompose(product["sales_quantity"], model='additive', period=52)
            seasonal_filled = res.seasonal.fillna(method='ffill')

            product["sales_quantity_diff"] = product["sales_quantity"].diff()
            product["sales_quantity"] = seasonal_filled
            normalized_sales_quantity = self.scaler.transform(product["sales_quantity"].values.reshape(-1, 1))
            normalized_sales_quantity = np.around(normalized_sales_quantity * 10) / 10

            # Convert the normalized numpy array back to Series
            normalized_series = pd.Series(normalized_sales_quantity.flatten(), index=product.index)
            # Add the normalized series to the list
            product["sales_quantity"] = normalized_series
            products_reshaped.append(product)
        self.scaled_products = products_reshaped
