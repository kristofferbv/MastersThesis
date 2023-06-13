import random
import sys

import matplotlib.pyplot as plt
import config_utils
import generate_data
import os
import signal
import retrieve_data
import numpy as np
from keras import layers, regularizers, Sequential
import tensorflow as tf
from keras.layers import Activation, Conv1D
from pandas import read_csv
import numpy as np
from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error

from MIP.analysis.analyse_data import plot_sales_quantity
from reinforcement_learning.environment import JointReplenishmentEnv

std_dev = 1
# Learning rate for actor-critic models
critic_lr = 0.003
actor_lr = 0.00001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.0005

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.1, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, state_shape, action_shape, buffer_capacity=10000000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.state_shape = state_shape
        self.action_shape = action_shape
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, *state_shape))
        self.action_buffer = np.zeros((self.buffer_capacity, action_shape))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *state_shape))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1
    def reset(self):
        self.state_buffer = np.zeros((self.buffer_capacity, *self.state_shape))
        self.action_buffer = np.zeros((self.buffer_capacity, self.action_shape))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, *self.state_shape))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-4)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, num_features, dtype=tf.float32):
        super(PositionalEncoding, self).__init__()
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, num_features, 2) * -(np.log(10000.0) / num_features))
        positional_encoding = np.zeros((sequence_length, num_features))
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        self.positional_encoding = tf.convert_to_tensor(positional_encoding, dtype=dtype)

    def call(self, inputs):
        return inputs + self.positional_encoding

class DDPG():
    def __init__(self, products, state_shape, env):
        self.env = env
        self.products = products
        self.state_shape = state_shape
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.std_dev = std_dev

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        self.buffer = Buffer(state_shape, len(products), 50000, 128)

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        signal.signal(signal.SIGINT, self.signal_handler)


    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)

        inputs = layers.Input(shape=self.state_shape)
        print(self.state_shape)
        out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.GRU(units=100, activation='relu', return_sequences=True)(out)
        out = layers.Dropout(rate=0.5)(out)
        out = layers.GRU(units=100, activation='relu')(out)
        # out = layers.Dropout(rate=0.5)(out)

        outputs = layers.Dense(len(self.products),  kernel_initializer=last_init)(out)

        # Multiply with action upper bound
        # outputs = outputs * 100
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.state_shape)
        state_out = layers.Dense(32, activation="relu")(state_input)
        state_out = layers.Flatten()(state_out)
        # Action as input
        action_input = layers.Input(shape=(len(self.products), 1))
        action_out = layers.Dense(32, activation="relu")(action_input)
        action_out = layers.Flatten()(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(concat)
        out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)

        out = layers.Flatten()(out)
        outputs = layers.Dense(1)(out)
        outputs = outputs
        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(self, state, noise_object, should_include_noise = True):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        if should_include_noise:
            sampled_actions = sampled_actions.numpy() + noise * 100
        else:
            sampled_actions = sampled_actions.numpy()
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, 0, 100)
        # discrete_actions =  np.round(legal_action / 2.5).astype(int)

        return [np.squeeze(legal_action)]

        # We compute the loss and update parameters

    def signal_handler(self, sig, frame):
        print('Training interrupted. Saving models...')
        self.save_models()
        self.plot_rewards()
        print('Models saved and rewards plotted. Exiting...')
        sys.exit(0)
    def plot_rewards(self):
        plt.plot(self.avg_reward_list)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.title('Costs as a function of epochs')
        plt.show()
    def save_models(self):
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        print('Saving models...')
        self.actor_model.save(f'actor_model')

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        if self.buffer.buffer_counter>1000000: # and random.random() > 0.9:
            priorities = np.arange(record_range)  # Use indices as priorities (age-based prioritization)
            probabilities = priorities / np.sum(priorities)  # Compute probabilities proportional to priorities
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.buffer.batch_size, p= probabilities)
        else:
            batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.abs(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        critic_grad, _ = tf.clip_by_global_norm(critic_grad, 1)
        critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        actor_grad, _ = tf.clip_by_global_norm(actor_grad, 0.01)  # Apply gradient clipping
        actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def train(self, should_plot=True):
        # hei = tf.keras.models.load_model("actor_model")
        # hade =  tf.keras.models.load_model("actor_model")
        # self.actor_model = hei
        # self.target_actor = hade


        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        self.avg_reward_list = []
        self.env.set_costs(self.products)
        epsilon = 0.5  # start with full randomness
        epsilon_min = 0.01  # the lowest level of randomness we want
        epsilon_decay = 0.995  # how quickly to decrease randomness
        # Takes about 4 min to train

        for ep in range(total_episodes):
            if ep > 100:
                actor_optimizer.learning_rate = 1e-3  # increased learning rate
            self.ep = ep
            if (ep > 380):
                self.env.set_costs(self.products)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            generated_products = generate_data.generate_seasonal_data_based_on_products(self.products, 500, period=52)


            self.env.products = generated_products
            prev_state = self.env.reset()
            self.env.reset_inventory()
            episodic_reward = 0
            prev_state = tf.convert_to_tensor([prev_state])
            prev_state = tf.transpose(prev_state, perm=[0,2, 1])


            while True:
                self.std_dev = self.std_dev * 0.999
                if (self.std_dev < 0.3):
                    self.std_dev = 0.3

                ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
                action = self.policy(prev_state, ou_noise)[0]

                for i in range(len(action)):
                    if action[i] < 1:
                        action[i] = 0
                # if random.random() <0.001:
                #     action = [0 for i in range(len(self.products))]
                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)
                state = tf.convert_to_tensor([state])
                # state = tf.reshape(state, [1, 13, -1])
                state = tf.transpose(state, perm=[0,2,1])
                total_reward = sum(reward)
                reward = sum(reward)
                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += total_reward

                # End this episode when `done` is True
                self.learn()
                self.update_target(self.target_critic.variables, self.critic_model.variables, tau)
                # if ep > 50:
                self.update_target(self.target_actor.variables, self.actor_model.variables, tau)
                if ep+1 % 50 == 0:
                    print("EP49Ac", action)
                if done:
                    print(action)
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)
            self.avg_reward_list.append(episodic_reward)

            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-30:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
        # self.actor_model.save(f'actor_model_ep{ep}_saved_model')
        if should_plot:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()
        self.test(episodes = 1)

    def test(self, episodes = 10, path = None):
        # loading model
        # actor = self.actor_model
        actor = tf.keras.models.load_model(f"actor_model_ep{99}_saved_model")
        avg_reward_list = []
        for episode in range(episodes):
            print(f"episode: {episode}")
            self.env.set_costs(self.products)

            episodic_reward = 0
            generated_products = generate_data.generate_seasonal_data_based_on_products(self.products, 500)
            self.env.products = generated_products
            prev_state = self.env.reset()
            prev_state = tf.transpose(prev_state, perm=[1,0])

            while True:
                tf_prev_state = tf.convert_to_tensor([prev_state])
                # std_dev  = 0.3
                # ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
                # action = self.policy(tf_prev_state, ou_noise, actor)[0]


                action = tf.squeeze(actor(tf_prev_state, training=False)).numpy()
                for i in range(len(action)):
                    if action[i] < 1:
                        action[i] = 0
                print(action)
                state, reward, done, info = self.env.step(action)
                total_reward = sum(reward)
                episodic_reward += total_reward

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state
                prev_state = tf.transpose(prev_state, perm=[1,0])
                # prev_state = tf.reshape(prev_state, [13, 8])

            avg_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
        print("Avg Reward is ==> {}".format(sum(avg_reward_list)/len(avg_reward_list)))
