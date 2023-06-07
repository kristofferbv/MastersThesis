import json

import numpy as np
import tensorflow as tf
from keras import layers, models
import keras
from keras.layers import Dense

import random

import numpy as np
import tensorflow as tf
from keras import layers, models

from generate_data import generate_seasonal_data_based_on_products

# Define hyperparameters
gamma = 0.98  # discount factor
tau = 0.005  # target network update rate
actor_lr = 0.00003  # learning rate of actor network
critic_lr = 0.001  # learning rate of critic network
buffer_capacity = 20000 # replay buffer capacity
batch_size = 64  # minibatch size
num_episodes = 1000
num_agents = 4  # number of agents
warm_up_steps = 1000


class Actor(models.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = layers.LSTM(64, return_sequences=True, activation='tanh')
        self.l2 = layers.LSTM(64, return_sequences=True, activation='tanh')
        self.l3 = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
        self.max_action = max_action

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return self.max_action * abs(x)

class Critic(models.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.l1 = layers.LSTM(32, return_sequences=True, activation='tanh')
        self.l2 = layers.LSTM(32, return_sequences=True, activation='tanh')
        self.l3 = layers.TimeDistributed(layers.Dense(1))
        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        return tf.reduce_mean(x, axis=1)


class Agent:
    def __init__(self, state_dim, action_dim, max_action, env, discount=0.99, tau=tau):
        self.env = env
        self.num_episodes = num_episodes
        self.actor = Actor(action_dim, max_action)
        self.actor_target = Actor(action_dim, max_action)
        self.actor_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=actor_lr)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau

        # self.update_network_parameters(tau=1)  # hard update for initialization

    def learn(self, replay_buffer, agents, agent_num, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        state = np.array(state, dtype=np.float32)
        # Selecting the next action for all agents according to their Target Actors
        next_actions = [agents[i].actor_target(next_state[:, i, :]) for i in range(len(agents))]
        next_actions = [tf.reshape(a, (-1,)) for a in next_actions]
        next_actions = tf.stack(next_actions, axis=1)

        # Compute the target Q value
        next_actions = tf.expand_dims(next_actions, axis=-1)
        action = tf.expand_dims(action, axis=-1)
        inputs = tf.concat([next_state, next_actions], axis=2)
        target_Q = self.critic_target(inputs)

        not_done = tf.cast(not_done, tf.float32)
        not_done = tf.reshape(not_done, (batch_size, 1))
        reward = reward.astype(np.float32)
        target_Q = tf.reshape(target_Q, [batch_size, 1])
        target_Q = reward[:, agent_num] + (not_done * self.discount * target_Q)

        # Compute critic loss
        with tf.GradientTape() as tape:
            inputs = tf.concat([state, action], axis=2)
            current_Q = self.critic(inputs)
            critic_loss = tf.reduce_mean(tf.square(current_Q - target_Q))
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # gradients, _ = tf.clip_by_global_norm(gradients, 0.8)  # Apply gradient clipping
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Compute actor loss
        # Compute actor loss
        with tf.GradientTape() as tape:
            current_actions = [agents[i].actor(state[:, i, :]) for i in range(len(agents))]
            current_actions = [tf.reshape(a, (-1,)) for a in current_actions]
            current_actions = tf.stack(current_actions, axis=1)
            current_actions = tf.expand_dims(current_actions, axis=-1)
            inputs = tf.concat([state, current_actions], axis=2)
            actor_loss = -self.critic(inputs)
            actor_loss = tf.reduce_mean(actor_loss)

            # Explicitly state that we want to watch the actor's variables
            tape.watch(self.actor.trainable_variables)

        # Optimize the actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # actor_grads, _ = tf.clip_by_global_norm(actor_grads, 5)  # Apply gradient clipping
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update the frozen target models
        self.update_target_networks()

    @tf.function
    def train_value_function(self, a, b, n_agents):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            critic_losses = [tf.keras.losses.MSE(a[index], b[index]) for index in range(n_agents)]
        value_grads = tape.gradient(critic_losses, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        for a, b in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
            weights.append(tau * b + (1 - tau) * a)
        self.actor_target.set_weights(weights)

        weights = []
        for a, b in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            weights.append(tau * b + (1 - tau) * a)
        self.critic_target.set_weights(weights)

    def update_target_networks(self, tau= None):
        if tau is None:
            tau = self.tau
        actor_weights = self.actor.weights
        target_actor_weights = self.actor_target.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.actor_target.set_weights(target_actor_weights)

        critic_weights = self.critic.weights
        target_critic_weights = self.critic_target.weights

        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.critic_target.set_weights(target_critic_weights)

    def select_action(self, state):
        state = tf.convert_to_tensor([state])
        print("UAJAJJAJAJAJ",self.actor(state))
        return self.actor(state)[0].numpy()


class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size=batch_size):
        priorities = np.arange(len(self.storage))  # Use indices as priorities (age-based prioritization)
        probabilities = priorities / np.sum(priorities)  # Compute probabilities proportional to priorities
        indices = np.random.choice(len(self.storage), size=batch_size, p=probabilities)  # Sample indices with probabilities
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for index in indices:
            state, action, next_state, reward, done = self.storage[index]
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            next_states.append(np.array(next_state, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)




class MultiAgent:
    def __init__(self, agents, env, real_products):
        self.products = real_products
        self.env = env
        self.num_episodes = num_episodes
        self.agents = agents
        self.replay_buffer = ReplayBuffer(20000)

    def train(self):
        # Main training loop
        # Warm-up phase
        state = self.env.reset()
        self.env.set_costs(self.products)
        # for _ in range(warm_up_steps):
        #     action = env.action_space.sample()  # Take random action
        #     next_state, reward, done, *_ = env.step(action)
        #     replay_buffer.add((state, action, reward, next_state, done))
        #     state = next_state
        #     if done:
        #         state = env.reset()

        # Main training loop
        running_avg_reward = 0
        running_std_reward = 1  # Initialize to 1 to avoid division by zero issues
        for episode in range(num_episodes):
            generate_seasonal_data_based_on_products(self.products, 500)
            done = False
            total_reward = 0
            state = self.env.reset()
            print("STAAATE", state)
            factor = 0.4
            samples = []
            while not done:
                if episode > 100:
                    factor *= 0.95
                    if factor < 0.2:
                        factor = 0.2
                # Select action according to policy
                if random.random() < factor:
                    actions = tf.random.uniform(shape=[4],minval=0,maxval=70)
                else:
                    # random_number = random.uniform(-1, 1)
                # actions = [agent.select_action(state[i])[0] + random_number * faktor for i, agent in enumerate(self.agents)]
                    actions = [np.clip(agent.select_action(state[i]), 0, 100) for i, agent in enumerate(self.agents)]
                    print("ACTIons", actions)
                # print("actions", actions)
                # Perform action and get reward
                next_state, reward, done, *_ = self.env.step(actions)
                total_reward += sum(reward)


                # Store experience in replay buffer
                self.replay_buffer.add((state, actions, next_state, reward, done))

                # Move to next state
                state = next_state

                 # Train agent
                if episode > 100:
                    if episode % 300 == 0:
                        for agent_num in range(len(self.agents)):
                            self.agents[agent_num].learn(self.replay_buffer, self.agents, agent_num, batch_size = len(self.replay_buffer.storage))
                    else:
                        for agent_num in range(len(self.agents)):
                            self.agents[agent_num].learn(self.replay_buffer, self.agents, agent_num, batch_size = batch_size)
            if episode > 100 or episode % 10 == 0:
                print(actions)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights('actor.h5')
        self.actor_target.save_weights('target_actor.h5')
        self.critic.save_weights('critic.h5')
        self.critic_target.save_weights('target_critic.h5')

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights('actor.h5')
        self.actor_target.load_weights('target_actor.h5')
        self.critic.load_weights('critic.h5')
