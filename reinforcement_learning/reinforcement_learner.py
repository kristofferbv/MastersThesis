import os

import pandas as pd

import config_utils
import generate_data
import retrieve_data
from reinforcement_learning.environment import JointReplenishmentEnv
from actor import *
from critic import *
from a2c_agent import *
from ppo import *
from generate_data import generate_seasonal_data_based_on_products
from maddpg import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

# Main function
if __name__ == "__main__":
    config = config_utils.load_config("config.yml")
    should_generate_data = config["rl"]["generate_data"]
    method = config["rl"]["method"]

    # TODO! Note to self: Maybe draw random inventory levels at start of each simulation?

    # If you want each group as a Series of 'transaction_amount', you can do:
    real_products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
    real_products = [df["sales_quantity"] for df in real_products][:4]
    products = real_products
    if should_generate_data:
        generated_products = generate_seasonal_data_based_on_products(real_products, 300)
        products = generated_products

    # Set up the environment
    env = JointReplenishmentEnv(products)
    # state_shape is the input shape for critic and actor
    state_shape = env.observation_space.shape[1]
    # Since agent space only consist of the state of a singe product
    state_shape_agent = state_shape
    # action shape is the output shape of the actor
    action_shape = env.action_space.n
    print(state_shape)
    print(action_shape)

    # set up the networks
    # actor = Actor(state_shape_agent, action_shape)
    # critic = Critic(state_shape)
    method = ""
    if method == "ppo":
        ppo_model = PPO(env, real_products)
        ppo_model.train_ppo()
        ppo_model.test(208)
    else:
        agents = []
        print("state", state_shape)
        print("action", action_shape)
        for product in products:
            agents.append(Agent((state_shape,len(products)), action_shape, 70, env, discount=0.99, tau=0.005))
        ma = MultiAgent(agents, env, real_products)
        ma.train()

    # else:
    #     # Train the A2C model
    #     a2c_model = A2CAgent(actor, critic, env)
    #     a2c_model.train_a2c()
