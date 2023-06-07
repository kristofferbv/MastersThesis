import os
import signal
import sys
import random

import matplotlib.pyplot as plt
from keras.models import load_model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gym
import scipy.signal
import actor as a
import time
from generate_data import generate_seasonal_data_based_on_products

import generate_data
import retrieve_data
from reinforcement_learning.environment import JointReplenishmentEnv

# Hyperparameters of the PPO algorithm
steps_per_epoch = 3000
epochs = 1000
gamma = 0.99
clip_ratio = 0.6
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (32, 64, 32)
exploration_rate = 0.1



def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros((size, observation_dimensions[0]), dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros((size, observation_dimensions[0]), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = np.array(action)
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = np.array(logprobability)[0]
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Flatten the input if it's not already flat
    x = keras.layers.Flatten()(x)

    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = keras.layers.Dense(units=size, activation=activation)(x)

    return keras.layers.Dense(units=sizes[-1], activation=output_activation)(x)


class PPO:
    def __init__(self, env, products):
        self.env = env
        self.products = products
        # Initialize the environment and get the dimensionality of the observation space and the number of possible actions
        self.observation_dimensions = env.observation_space.shape
        print("OBBS", self.observation_dimensions)
        self.num_actions = env.action_space.n
        self.exploration_rate = exploration_rate

        # Initialize the buffer
        self.buffer = Buffer(env.observation_space.shape, steps_per_epoch)

        # Initialize the actor for all products and the critic as keras models
        self.observation_input_actor = keras.Input(shape=self.observation_dimensions[1], dtype=tf.float32)
        self.observation_input_critic = keras.Input(shape=self.observation_dimensions, dtype=tf.float32)


        logits = mlp(self.observation_input_actor, list(hidden_sizes) + [self.num_actions], tf.tanh, None)
        self.actors = []
        for i in range(len(products)):
            self.actors.append(keras.Model(inputs=self.observation_input_actor, outputs=logits))

        value = tf.squeeze(mlp(self.observation_input_critic, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
        print(self.observation_input_critic)
        self.critic = keras.Model(inputs=self.observation_input_critic, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)

        # List to store rewards for each epoch
        self.rewards_per_epoch = []

    # Initialize the observation, episode return and episode length
    # observation = observation[0]

    def train_ppo(self):
        observation, episode_return, episode_length = self.env.reset(), 0, 0
        self.env.set_costs(self.products)

        # Iterate over the number of epochs
        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0
            should_decay = False


            # Iterate over the steps of each epoch
            for t in range(steps_per_epoch):
                # Get the logits, action, and take one step in the environment
                logits = []
                actions = []
                for i, obs in enumerate(observation):
                    logit, action = self.sample_action(obs, i, should_decay)
                    should_decay = False
                    logits.append(logit)
                    actions.append(action[0].numpy())
                individual_actions = actions

                # Unwrap the observation from a tuple to a list before feeding it to the environment
                observation_new, reward, done, *_ = self.env.step(individual_actions)
                reward = sum(reward)
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.critic(tf.reshape(observation, [1, *observation.shape]))
                logprobability_t = [self.logprobabilities(logit, act) for logit, act in zip(logits, actions)]

                # Store obs, act, rew, v_t, logp_pi_t
                self.buffer.store(observation, actions, reward, value_t, logprobability_t)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(np.expand_dims(observation, axis=0))
                    self.buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = self.env.reset(), 0, 0
                    self.env.products = generate_seasonal_data_based_on_products(self.products, 500)

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(train_policy_iterations):
                for i in range(len(self.products)):
                    observation_i = observation_buffer[:, i, :]
                    action_i = action_buffer[:, i]
                    logprobability_i = logprobability_buffer[:, i]

                    kl = self.train_policy(observation_i, action_i, logprobability_i, advantage_buffer, i)
                    if kl > 1.5 * target_kl:
                        # Early Stopping
                        break

                # kl = self.train_policy(
                #     observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                # )
                # if kl > 1.5 * target_kl:
                #     # Early Stopping
                #     break

            # Update the value function
            for _ in range(train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

            # Append the mean reward per epoch to the list
            mean_reward = sum_return / num_episodes
            self.rewards_per_epoch.append(mean_reward)

            # Print mean return and length for each epoch
            print(
                f" Epoch: {epoch + 1}. Mean Reward: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes} Current_time: {self.env.time_period}"
            )

        self.save_models()
        self.plot_rewards()

    def test(self, start_time_period):
        # Want to print the environment, costs and actions:
        self.env.verbose = True
        generate_seasonal_data_based_on_products(self.products, 500)
        self.actors = []
        # Load the actor networks
        for i in range(len(self.products)):
            self.actors.append(load_model(os.path.join('models', f'actor_model_{i}.h5')))

        done = False
        total_costs = 0
        self.env.time_period = start_time_period
        observation, episode_return, episode_length = self.env.reset(), 0, 0
        for i in range(0,14):
            print(f"Inventory_level at start of period {self.env.current_period}: {self.env.inventory_levels}")
            # Get the logits, action, and take one step in the environment
            # observation = observation.reshape(1, -1)
            logits = []
            actions = []
            for index, obs in enumerate(observation):
                obs = np.array(tf.reshape(obs, [-1]))
                obs = np.expand_dims(obs, axis=0)  # shape is now (1, shape)
                logit = self.actors[index](obs)
                action = tf.argmax(logit, axis=1)
                logits.append(logit)
                actions.append(action[0].numpy())
            individual_actions = actions

            # logits = self.actor.predict(observation)
            # action = tf.argmax(logits, axis=1)

            # individual_actions = a.unflatten_action(action.numpy()[0], self.env.action_space.n, len(self.products))
            print(f"Action for time period {self.env.current_period}: {individual_actions}")
            # Just a consequence of the epoch hack
            individual_actions.append(1)
            observation_new, reward, done, *_ = self.env.step(individual_actions)
            demand = []
            # for product in self.products:
            #     current_period = self.env.current_period
            #     demand.append(product.iloc[current_period])
            # print(f"Demand for time period {self.env.current_period}: {demand}")
            # print(f"Reward for time period {self.env.current_period}: {reward}")

            total_costs += -reward
            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = i == 13
            if terminal:
                print("Total_costs: ", total_costs)

    def plot_rewards(self):
        plt.plot(self.rewards_per_epoch)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Costs as a function of epochs')
        plt.show()

    def save_models(self):
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        print('Saving models...')
        for i, actor in enumerate(self.actors):
            actor.save(os.path.join(save_dir, f'actor_model_{i}.h5'))
        self.critic.save(os.path.join(save_dir, 'critic_model.h5'))

    def signal_handler(self, sig, frame):
        print('Training interrupted. Saving models...')
        self.save_models()
        self.plot_rewards()
        print('Models saved and rewards plotted. Exiting...')
        sys.exit(0)


    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

        # Sample action from actor


    @tf.function
    def sample_action(self, observation, index, should_decay):
        observation = tf.reshape(observation, [1, *observation.shape])
        logits = self.actors[index](observation)
        random_number = random.random()
        should_decay = False
        if should_decay:
            self.exploration_rate *= 0.97
        if random_number < self.exploration_rate:
            action = tf.random.categorical(tf.math.log([[1 / logits.shape[-1]] * logits.shape[-1]]), 1)[0]
        else:
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, i):
        with tf.GradientTape(persistent=True) as tape:  # Record operations for automatic differentiation.
            # reshape the observation_buffer to (batch_size * n_products, n_features)
            # print(observation_buffer)
            # print(action_buffer)
            # print(logprobability_buffer)
            reshaped_observation_buffer = tf.reshape(observation_buffer, (-1, self.observation_dimensions[1]))

            # apply the actor network to the batch of product states
            logits = self.actors[i](reshaped_observation_buffer)

            # reshape the logits back to (batch_size, n_products, -1)
            # logits = tf.reshape(logits, (-1, self.observation_dimensions[0], logits.shape[-1]))
            # action_buffer = tf.reshape(action_buffer, (-1,))
            # logprobability_buffer = tf.reshape(logprobability_buffer, (-1,))

            # advantage_buffer = tf.expand_dims(advantage_buffer, axis=-1)
            # advantage_buffer = tf.tile(advantage_buffer, [1, len(self.products)])
            # advantage_buffer = tf.reshape(advantage_buffer, [-1])

            # print("action_buffer: ", action_buffer)
            # print("logits: ", logits)
            # print("logprobability buffer: ", logprobability_buffer)
            # print()
            ratio = tf.exp(
                self.logprobabilities(logits, action_buffer) - logprobability_buffer
            )

            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,
            )


            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
        for actor in self.actors:
            policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

        kl = tf.reduce_mean(logprobability_buffer - self.logprobabilities(logits, action_buffer))
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))



