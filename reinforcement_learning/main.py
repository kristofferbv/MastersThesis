import numpy as np
import tensorflow as tf
import keras
import gym

class JRPEnv(gym.Env):
    def __init__(self):
        self.demand_mean = 10 # Mean demand
        self.demand_std = 3 # Standard deviation of demand
        self.unit_cost = 1 # Cost of ordering one unit of inventory
        self.holding_cost = 0.1 # Cost of holding one unit of inventory
        self.max_inventory = 20 # Maximum inventory level
        self.min_inventory = 0 # Minimum inventory level
        self.inventory = None # Current inventory level
        self.time_step = 0 # Current time step
        self.action_space = gym.spaces.Discrete(self.max_inventory+1) # Action space
        self.observation_space = gym.spaces.Discrete(self.max_inventory+1) # Observation space

    def step(self, action):
        demand = np.random.normal(self.demand_mean, self.demand_std)
        reward = -self.unit_cost * float(action) - self.holding_cost * self.inventory # Make sure action is a scalar
        self.inventory = max(min(self.inventory + action - demand, self.max_inventory), self.min_inventory)
        done = False
        self.time_step += 1
        if self.time_step == 100: # Terminal state after 100 time steps
            done = True
        return self.inventory, reward, done, {}

    def reset(self):
        self.inventory = self.max_inventory // 2 # Start with half of the maximum inventory
        self.time_step = 0
        return self.inventory

class ActorCritic:
    def __init__(self, env):
        self.env = env
        self.actor_lr = 0.001 # Actor learning rate
        self.critic_lr = 0.01 # Critic learning rate
        self.gamma = 0.99 # Discount factor
        self.actor = self.build_actor() # Actor network
        self.critic = self.build_critic() # Critic network
        self.rewards = [] # List to store total rewards for each episode

    def build_actor(self):
        actor = keras.models.Sequential()
        actor.add(keras.layers.Dense(32, activation='relu', input_shape=(1,)))
        actor.add(keras.layers.Dense(self.env.action_space.n, activation='softmax'))
        actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.actor_lr), loss='categorical_crossentropy')
        return actor

    def build_critic(self):
        critic = keras.models.Sequential()
        critic.add(keras.layers.Dense(32, activation='relu', input_shape=(1,)))
        critic.add(keras.layers.Dense(1))
        critic.compile(optimizer=keras.optimizers.Adam(learning_rate=self.critic_lr), loss='mse')
        return critic

    def choose_action(self, state):
        state = np.reshape(state, (1, 1)) # Res

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env


env = JRPEnv()
agent = ActorCritic(env)

agent.train(num_episodes=1000)

num_episodes = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print("Episode {}: Total reward = {}".format(episode+1, total_reward))

