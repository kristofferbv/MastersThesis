import os
import signal
import sys

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
steps_per_epoch = 4000
epochs = 3000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (128, 128)


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
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
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
        self.num_actions = env.action_space.n ** len(products)

        # Initialize the buffer
        self.buffer = Buffer(self.observation_dimensions, steps_per_epoch)

        # Initialize the actor and the critic as keras models
        self.observation_input = keras.Input(shape=self.observation_dimensions, dtype=tf.float32)
        self.logits = mlp(self.observation_input, list(hidden_sizes) + [self.num_actions], tf.tanh, None)
        print(self.logits)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)

        value = tf.squeeze(
            mlp(self.observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        self.critic = keras.Model(inputs=self.observation_input, outputs=value)

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

        # Iterate over the number of epochs
        for epoch in range(epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0
            sum_length = 0
            num_episodes = 0

            # Iterate over the steps of each epoch
            for t in range(steps_per_epoch):
                # Get the logits, action, and take one step in the environment
                # observation = observation.reshape(1, -1)
                logits, action = self.sample_action(observation)
                individual_actions = a.unflatten_action(action[0].numpy(), self.env.action_space.n, len(self.products))
                individual_actions.append(epoch)
                observation_new, reward, done, *_ = self.env.step(individual_actions)
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.critic(tf.reshape(observation, [1, *observation.shape]))
                logprobability_t = self.logprobabilities(logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                self.buffer.store(observation, action, reward, value_t, logprobability_t)

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
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * target_kl:
                    # Early Stopping
                    break

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
            # if epoch != 0 and epoch % 145 == 0:
            #     self.env.reset_time_period()
            # if epoch != 0 and epoch % 3 == 0:
            #     self.env.increase_time_period(5)

        self.save_models()
        self.plot_rewards()

    def test(self, start_time_period):
        generate_seasonal_data_based_on_products(self.products, 500)
        actor_model_dir = 'models/actor_model.h5'
        # Load the actor network
        self.actor = load_model(actor_model_dir)

        done = False
        total_costs = 0
        self.env.time_period = start_time_period
        observation, episode_return, episode_length = self.env.reset(), 0, 0
        for i in range(0,53):
            print(f"Inventory_level at start of period {self.env.current_period}: {self.env.inventory_levels}")
            # Get the logits, action, and take one step in the environment
            # observation = observation.reshape(1, -1)
            observation = tf.reshape(observation, [1, *observation.shape])
            logits = self.actor.predict(observation)
            action = tf.argmax(logits, axis=1)
            print(action)

            individual_actions = a.unflatten_action(action.numpy()[0], self.env.action_space.n, len(self.products))
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
            terminal = i == 52
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
        self.actor.save(os.path.join(save_dir, 'actor_model.h5'))
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


    def sample_action(self, observation):
        observation = tf.reshape(observation, [1, *observation.shape])
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action


    # Train the policy by maxizing the PPO-Clip objective
    @tf.function
    def train_policy(
            self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + clip_ratio) * advantage_buffer,
                (1 - clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl


    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


