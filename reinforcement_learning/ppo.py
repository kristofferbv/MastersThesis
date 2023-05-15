import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gym
import scipy.signal
import actor as a
import time

import generate_data_dataframe
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
hidden_sizes = (64, 64)



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
        self.num_actions = env.action_space.n**len(products)

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
                individual_actions = a.unflatten_action(action[0].numpy(), self.env.action_space.n,len(self.products))
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
                    # observation = observation[0]

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

            # Print mean return and length for each epoch
            print(
                f" Epoch: {epoch + 1}. Mean Reward: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
            )

    def logprobabilities(self,logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability

    # Sample action from actor
    @tf.function
    def sample_action(self,observation):
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
