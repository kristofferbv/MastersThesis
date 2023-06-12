import os

import pandas as pd

import config_utils
import generate_data
import retrieve_data
from MIP.analysis.analyse_data import plot_sales_quantity
from reinforcement_learning.ddpg2 import DDPG
from reinforcement_learning.environment import JointReplenishmentEnv
from actor import *
from critic import *
from a2c_agent import *
#from ppo import *
from generate_data import *
#from maddpg import *
#from ddpg2 import *
#from maddpg import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

# Main function
if __name__ == "__main__":
    config = config_utils.load_config("config.yml")
    should_generate_data = config["rl_model"]["generate_data"]
    method = config["rl_model"]["method"]
    seed = config["main"]["seed"]
    product_categories = config["rl_model"]["product_categories"]


    real_products = []
    if seed is not None:
        # Setting a random seed ensure we select the same random products each time
        random.seed(seed)
    for category in product_categories.keys():
        category_products = retrieve_data.read_products("2016-01-01", "2020-12-30", category)
        print(category)
        number_of_products = product_categories[category]
        print(number_of_products)
        if number_of_products > 0:
            hei = random.sample(category_products, product_categories[category])
            if category == "erratic":
                plot_sales_quantity(generate_seasonal_data_for_erratic_demand(hei, 260))
            elif category == "smooth":
                plot_sales_quantity(generate_seasonal_data_for_erratic_demand(hei, 260))
            else:
                plot_sales_quantity(generate_seasonal_data_for_intermittent_demand(hei, 260))

            real_products += hei

    for product in real_products:
        print(product["product_hash"].iloc[1])

    # TODO! Note to self: Maybe draw random inventory levels at start of each simulation?

    # If you want each group as a Series of 'transaction_amount', you can do:
    # real_products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])[:4]
    # real_products = real_products[:4]
    # if should_generate_data:
    generated_products = generate_seasonal_data_based_on_products(real_products, 500)
    products = generated_products
    # plot_sales_quantity(products)
    # for i in generated_products:
    #     print(i["sales_quantity"])
    # plot_sales_quantity(real_products)
    # # print(product["product_hash"].iloc[100])
    #
    # Set up the environment
    env = JointReplenishmentEnv(products)
    # state_shape is the input shape for critic and actor
    state_shape = env.observation_space.shape
    # Since agent space only consist of the state of a singe product
    state_shape_agent = state_shape
    # action shape is the output shape of the actor
    action_shape = env.action_space.n
    #
    method = "ddpg"
    # if method == "ppo":
    #     ppo_model = PPO(env, real_products)
    #     ppo_model.train_ppo()
    #     ppo_model.test(208)
    #
    # elif method == "maddpg":
    #     agents = []
    #     print("state", state_shape)
    #     print("action", action_shape)
    #     for product in products:
    #         agents.append(Agent((state_shape,len(products)), action_shape, 100, env, discount=0.99, tau=0.005))
    #     ma = MultiAgent(agents, env, real_products)
    #     ma.train()
    #     ma.test()
    if method == "ddpg":
        ddpg = DDPG(real_products, state_shape, env)
        # ddpg.train()
        ddpg.test()

