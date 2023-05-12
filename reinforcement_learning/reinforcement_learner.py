import os

import retrieve_data
from reinforcement_learning.environment import JointReplenishmentEnv
from actor import *
from critic import *
from a2c_agent import *

# Main function
if __name__ == "__main__":

    # TODO! Note to self: Maybe draw random inventory levels at start of each simulation?
    print(os.getcwd())
    # Load the products data
    products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
    products = products
    # Set up the environment and networks
    env = JointReplenishmentEnv(products)
    # state_shape is the input shape for critic and actor
    state_shape = env.observation_space.shape
    print(state_shape)
    # action shape is the output shape of the actor
    action_shape = env.action_space.shape

    actor = Actor(state_shape, action_shape)
    critic = Critic(state_shape)

    # Train the A2C model
    a2c_model = A2CAgent(actor, critic, env)
    a2c_model.train_a2c()
