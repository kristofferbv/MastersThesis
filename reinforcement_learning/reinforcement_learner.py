import os

import pandas as pd

import config_utils
import generate_data_dataframe
import retrieve_data
from reinforcement_learning.environment import JointReplenishmentEnv
from actor import *
from critic import *
from a2c_agent import *
from ppo import *

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the directory of the current script
os.chdir(current_dir)

# Main function
if __name__ == "__main__":
    config = config_utils.load_config("config.yml")
    should_use_real_data = config["rl"]["real_data"]
    method = config["rl"]["method"]

    # TODO! Note to self: Maybe draw random inventory levels at start of each simulation?

    # If you want each group as a Series of 'transaction_amount', you can do:
    should_use_real_products = True
    if should_use_real_products:
        products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
        products = [df["sales_quantity"] for df in products][:4]
    else:
        df = generate_data_dataframe.generate_data(2, weeks=204)
        # Group your DataFrame by 'ID' and convert each group to a DataFrame
        groups = [group for _, group in df.groupby('ID')]
        products = [group['transaction_amount'] for _, group in df.groupby('ID')]

    # Set up the environment and networks
    env = JointReplenishmentEnv(products)
    # state_shape is the input shape for critic and actor
    state_shape = env.observation_space.shape
    # Since agent space only consist of the state of a singe product
    state_shape_agent = state_shape[1]
    # action shape is the output shape of the actor
    action_shape = env.action_space.n

    actor = Actor(state_shape_agent, action_shape)
    critic = Critic(state_shape)
    if method == "ppo":
        ppo_model = PPO(env, products)
        ppo_model.train_ppo()

    else:
        # Train the A2C model
        a2c_model = A2CAgent(actor, critic, env)
        a2c_model.train_a2c()
