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

from MIP.analysis.analyse_data import plot_sales_quantity
from reinforcement_learning.environment import JointReplenishmentEnv

# # Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# # Change the working directory to the directory of the current script
# os.chdir(current_dir)
#
# # Main function
# config = config_utils.load_config("config.yml")
# should_generate_data = config["rl"]["generate_data"]
# method = config["rl"]["method"]
#
# # If you want each group as a Series of 'transaction_amount', you can do:
# real_products = retrieve_data.read_products_with_hashes("2016-01-10", "2020-12-30", ["569b6782ce5885fc4abf21cfde38f7d7", "92b1f191dfce9fff64b4effd954ccaab", "8ef91aac79542f11dedec4f79265ae3a", "2fa9c91f40d6780fd5b3c219699eb139", "1fb096daa569c811723ce8796722680e", "f7b3622f9eb50cb4eee149127c817c79"])
# real_products = real_products[:4]
# # real_products = [df["sales_quantity"] for df in real_products][:4]
# for product in real_products:
#     print(product)
# generated_products = generate_data.generate_seasonal_data_based_on_products(real_products, 300, 0)
# products = generated_products
#
# # Set up the environment
# env = JointReplenishmentEnv(products)
# # state_shape is the input shape for critic and actor
# state_shape = env.observation_space.shape
# print(state_shape)
# # Since agent space only consist of the state of a singe product
# state_shape_agent = state_shape
# # action shape is the output shape of the actor
# action_shape = env.action_space.n

# set up the networks
# actor = Actor(state_shape_agent, action_shape)
# critic = Critic(state_shape)


std_dev = 1
# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10001
# Discount factor for future rewards
gamma = 0.998
# Used to update target networks
tau = 0.0005

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
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
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

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

    # def get_actor(self):
    #     # Initialize weights between -3e-3 and 3-e3
    #     last_init = tf.random_uniform_initializer(minval=-0.001, maxval=0.001)
    #
    #     inputs = layers.Input(shape=self.state_shape)
    #     out = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    #     # model.add(LSTM(128, activation='relu'))
    #     # out = layers.Flatten()(out)
    #     out = layers.LSTM(units=32, return_sequences=False, activation="relu",kernel_initializer="lecun_normal")(out)  # Adjust the number of units to your preference
    #     out = layers.Dropout(rate=0.5)(out)
    #     out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(inputs)
    #     out = layers.Dropout(rate=0.2)(out)
    #     out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)
    #     out = layers.Dropout(rate=0.2)(out)
    #     # out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)
    #     # out = layers.Dropout(rate=0.5)(out)
    #     outputs = layers.Dense(1,  kernel_initializer=last_init)(out)
    #
    #     # Multiply with action upper bound
    #     # outputs = outputs * 100
    #     model = tf.keras.Model(inputs, outputs)
    #     return model

    # def get_actor(self):
    #     # Initialize weights between -3e-3 and 3-e3
    #     last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    #
    #     inputs = layers.Input(shape=(self.state_shape))
    #     out = layers.Dense(256, activation="tanh")(inputs)
    #     out = layers.Dense(256, activation="tanh")(out)
    #     outputs = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)(out)
    #
    #     # Our upper bound is 2.0 for Pendulum.
    #     outputs = outputs * 5
    #     model = tf.keras.Model(inputs, outputs)
    #     return model

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=self.state_shape)
        x = TransformerBlock(13, 8, 16)(state_input)  # Add an additional TransformerBlock layer
        state_out = layers.GlobalAveragePooling1D()(x)
        state_out = layers.Dense(32, activation="relu")(state_input)
        # state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(len(self.products), 1))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(concat)
        out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(out)

        out = layers.Flatten()(out)
        outputs = layers.Dense(1)(out)
        outputs = outputs * 100

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def get_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        inputs = layers.Input(shape=self.state_shape)
        x = TransformerBlock(13, 20, 32)(inputs)
        x = TransformerBlock(13, 12, 32)(x)  # Add an additional TransformerBlock layer
        x = TransformerBlock(13, 4, 32)(x)  # Add an additional TransformerBlock layer
        x = layers.GlobalAveragePooling1D()(x)
        # x = layers.Dense(16, activation="relu")(x)
        # x = layers.Dropout(rate=0.5)(x)
        outputs = layers.Dense(8, activation="sigmoid", kernel_initializer=last_init)(x)
        outputs = outputs * 100
        model = tf.keras.Model(inputs, outputs)
        return model

    # def get_critic(self):
    #     # State as input
    #     state_input = layers.Input(shape=self.state_shape)
    #     state_out = layers.Dense(16, activation="relu", kernel_initializer="lecun_normal")(state_input)
    #     # state_out = TransformerBlock(16, 2, 64)(state_out)
    #     # state_out = layers.GlobalAveragePooling1D()(state_out)
    #
    #     # Action as input
    #     action_input = layers.Input(shape=(len(self.products), 1))
    #     action_out = layers.Dense(32, activation="relu", kernel_initializer="lecun_normal")(action_input)
    #     # action_out = TransformerBlock(32, 4, 128)(action_out)
    #     # action_out = layers.GlobalAveragePooling1D()(action_out)
    #
    #     concat = layers.Concatenate()([state_out, action_out])
    #
    #     out = layers.Dense(256, activation="relu", kernel_initializer="lecun_normal")(concat)
    #     out = layers.Dense(256, activation="relu", kernel_initializer="lecun_normal")(out)
    #     # out = TransformerBlock(256, 8, 256)(out)
    #     # out = layers.GlobalAveragePooling1D()(out)
    #     outputs = layers.Dense(1)(out)
    #     outputs = outputs * 100
    #
    #     model = tf.keras.Model([state_input, action_input], outputs)
    #
    #     return model

    def policy(self, state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise * 100
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
        if self.buffer.buffer_counter>10000: # and random.random() > 0.9:
            # print("NEEEEEEWWWWW BATCH SIZE")
            # self.batch_size = 128
            # self.buffer.reset()
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
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

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
        actor_grad, _ = tf.clip_by_global_norm(actor_grad, 1)  # Apply gradient clipping
        actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    def train(self, should_plot=True):
        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        self.avg_reward_list = []
        self.env.set_costs(self.products, 100)
        epsilon = 1  # start with full randomness
        epsilon_min = 0.01  # the lowest level of randomness we want
        epsilon_decay = 0.995  # how quickly to decrease randomness
        # Takes about 4 min to train
        running_avg_reward = 0
        running_std_reward = 1
        for ep in range(total_episodes):
            if (ep > 380):
                self.env.set_costs(self.products, 1)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            generated_products = generate_data.generate_seasonal_data_based_on_products(self.products, 500, period=52)
            # print("plotting")
            # plot_sales_quantity(generated_products)

            self.env.products = generated_products
            self.env.scale_demand(generated_products)
            prev_state = self.env.reset()
            if ep<500:
                self.env.reset_inventory()
            episodic_reward = 0

            while True:
                tf_prev_state = tf.convert_to_tensor([prev_state])
                self.std_dev = self.std_dev * 0.999
                if (self.std_dev < 0.3):
                    self.std_dev = 0.3
                ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
                action = self.policy(tf_prev_state, ou_noise)[0]
                # if np.random.rand() <= epsilon:  # decide whether to explore or exploit
                #     # exploration: choose a random action
                #     new_actions = []
                #     for a in action:
                #         if np.random.rand() <= 0.5:
                #             new_actions.append(np.random.choice(50))
                #         else: new_actions.append(a)
                #     action = new_actions

                for i in range(len(action)):
                    if action[i] < 5:
                        action[i] = 0
                # if random.random() <0.001:
                #     action = [0 for i in range(len(self.products))]
                #     print("YEEEEEEEEEeeeeeeeeeeeeeeeSSSSS!!!!")
                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)
                total_reward = sum(reward)

                reward = sum(reward)
                # running_avg_reward = 0.99 * running_avg_reward + 0.01 * sum(reward) * 2
                # running_std_reward = np.sqrt(0.99 * running_std_reward ** 2 + 0.01 * (sum(reward) * 2 - running_avg_reward) ** 2)
                # reward = -abs((sum(reward) - running_avg_reward) / running_std_reward)

                self.buffer.record((prev_state, action, reward, state))
                episodic_reward += total_reward

                # End this episode when `done` is True
                self.learn()
                self.update_target(self.target_actor.variables, self.actor_model.variables, tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, tau)

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
        self.actor_model.save(f'actor_model_ep{ep}_saved_model')
        if should_plot:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()

    def test(self, episodes = 1, path="actor_model"):
        # loading model
        actor = tf.keras.models.load_model(path)
        avg_reward_list = []
        for episode in range(episodes):
            print(f"episode: {episode}")
            self.env.set_costs(self.products)

            episodic_reward = 0
            generated_products = generate_data.generate_seasonal_data_based_on_products(self.products, 500)
            self.env.products = generated_products
            self.env.scale_demand(generated_products)
            prev_state = self.env.reset()

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

            avg_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
        print("Avg Reward is ==> {}".format(sum(avg_reward_list)/len(avg_reward_list)))
