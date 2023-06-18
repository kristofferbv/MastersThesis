import os
import signal
import sys

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Conv1D
from keras.layers import Layer
from keras.models import Sequential
import random

from scipy import stats

import config_utils
import generate_data
import matplotlib as mpl

import matplotlib.font_manager
plt.rcParams["font.family"] = "CMU Concrete"

std_dev = 1
# Learning rate for actor-critic models
critic_lr = 0.003
actor_lr = 0.00001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 1000
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
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
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
    def __init__(self, products, state_shape, env, product_categories):
        self.env = env
        self.product_categories = product_categories
        self.products = products
        self.state_shape = state_shape
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()
        self.std_dev = std_dev

        config = config_utils.load_config("config.yml")
        self.should_reset_time_at_each_episode = config["environment"]["should_reset_at_each_episode"]

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

        outputs = layers.Dense(len(self.products), kernel_initializer=last_init)(out)

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

    def policy(self, state, noise_object, should_include_noise=True):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        if should_include_noise:
            sampled_actions = sampled_actions.numpy() + noise * 100
        else:
            sampled_actions = sampled_actions.numpy()
        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, 0, 500)
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
        # You need to have the CMU Concrete font installed on your system for this to work
        mpl.rc('font', family='CMU Concrete')

        plt.plot(np.abs(self.avg_reward_list))
        plt.xlabel('Episodes')
        plt.ylabel('Costs (NOK)')
        plt.title('Costs as a function of epochs')

        plt.savefig('plot.png', dpi=300)
        plt.show()

    def save_models(self):
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        print('Saving models...')
        self.actor_model.save(f'models/actor_model_2')
        self.critic_model.save(f'models/critic_model_2')

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        if self.buffer.buffer_counter > 1000000:  # and random.random() > 0.9:
            priorities = np.arange(record_range)  # Use indices as priorities (age-based prioritization)
            probabilities = priorities / np.sum(priorities)  # Compute probabilities proportional to priorities
            # Randomly sample indices
            batch_indices = np.random.choice(record_range, self.buffer.batch_size, p=probabilities)
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

    def train(self, should_plot=True, reward_interval=1):
        hei = tf.keras.models.load_model("actor_model_1")
        hade = tf.keras.models.load_model("actor_model_1")
        self.actor_model = hei
        self.target_actor = hade

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        self.avg_reward_list = []
        self.env.set_costs(self.products)
        epsilon = 0.5  # start with full randomness
        epsilon_min = 0.01  # the lowest level of randomness we want
        epsilon_decay = 0.995  # how quickly to decrease randomness
        generated_products = self.generate_products(5000)
        self.env.products = generated_products

        for ep in range(total_episodes):
            if ep > 20:
                actor_optimizer.learning_rate = 1e-4  # increased learning rate
            self.ep = ep
            if (ep > 380):
                self.env.set_costs(self.products)
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if self.should_reset_time_at_each_episode:
                generated_products = self.generate_products(500)
                self.env.products = generated_products
                self.env.scaled_products = generated_products
            else:
                if self.env.current_period < 4999:
                    self.env.reset_current_period()
                    generated_products = self.generate_products(5000)
                    self.env.products = generated_products
                    self.env.scaled_products = generated_products
            prev_state = self.env.reset()
            # self.env.reset_inventory()
            episodic_reward = 0
            prev_state = tf.convert_to_tensor([prev_state])
            prev_state = tf.transpose(prev_state, perm=[0, 2, 1])

            step_count = 0  # Initialize step count
            episodic_actions = []  # List to store actions during episode
            episodic_states = []  # List to store states during episode

            while True:
                self.std_dev = self.std_dev * 0.99
                if (self.std_dev < 0.3):
                    self.std_dev = 0.3

                ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
                action = self.policy(prev_state, ou_noise)[0]

                for i in range(len(action)):
                    if action[i] < 5:
                        action[i] = 0

                # Recieve state and reward from environment.
                state, reward, done, info = self.env.step(action)
                state = tf.convert_to_tensor([state])
                state = tf.transpose(state, perm=[0, 2, 1])
                total_reward = sum(reward)
                reward = sum(reward)
                episodic_reward += total_reward

                episodic_actions.append(action)
                episodic_states.append(prev_state)

                # Increase step count
                step_count += 1

                if step_count % reward_interval == 0:
                    for s, a in zip(episodic_states, episodic_actions):
                        self.buffer.record((s, a, reward, state))
                    episodic_states = []  # Reset states list
                    episodic_actions = []  # Reset actions list
                    self.learn()
                    self.update_target(self.target_critic.variables, self.critic_model.variables, tau)
                    # if ep > 20:
                    self.update_target(self.target_actor.variables, self.actor_model.variables, tau)

                # End this episode when `done` is True
                if ep + 1 % 50 == 0:
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
        self.actor_model.save(f'models/actor_model_ep{ep}_saved_model')
        self.critic_model.save(f'models/critic_model_ep{ep}_saved_model')

        if should_plot:
            # Plotting graph
            # Episodes versus Avg. Rewards
            plt.plot(avg_reward_list)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()
        self.test(episodes=1)

    def test(self, episodes=1000, path=None):
        # loading model
        # actor = self.actor_model_training
        actor = tf.keras.models.load_model(f'models/beating_MIP')
        generated_products = self.generate_products(6000,0)
        self.env.products = generated_products
        self.env.scaled_products = generated_products
        achieved_service_level = {}

        avg_reward_list = []
        for episode in range(episodes):
            print(f"episode: {episode}")
            self.env.set_costs(self.products)
            if self.should_reset_time_at_each_episode:
                generated_products = self.generate_products(500)
                self.env.products = generated_products
                self.env.scaled_products = generated_products
            elif self.env.current_period < 4999:
                self.env.reset_current_period()
                generated_products = self.generate_products(5000)
                self.env.products = generated_products
                self.env.scaled_products = generated_products
            episodic_reward = 0
            prev_state = self.env.reset()
            prev_state = tf.transpose(prev_state, perm=[1, 0])

            while True:
                tf_prev_state = tf.convert_to_tensor([prev_state])
                # std_dev  = 0.3
                # ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
                # action = self.policy(tf_prev_state, ou_noise, actor)[0]

                action = tf.squeeze(actor(tf_prev_state, training=False)).numpy()
                # action = [random.randint(0,80) for i in range(2)]
                for i in range(len(action)):
                    if action[i] < 10:
                        action[i] = 0
                # print(action)
                state, reward, done, info = self.env.step(action)
                for i in range(len(self.products)):
                    if i not in achieved_service_level.keys():
                        achieved_service_level[i] = []
                    achieved_service_level[i].append(self.env.achieved_service_level[i])
                total_reward = sum(reward)
                episodic_reward += total_reward

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state
                prev_state = tf.transpose(prev_state, perm=[1, 0])
                # prev_state = tf.reshape(prev_state, [13, 8])
            if episodes!=0:
                avg_reward_list.append(episodic_reward)
        se = stats.sem(avg_reward_list)
        confidence = 0.95
        ci = se * stats.t.ppf((1 + confidence) / 2., len(avg_reward_list) - 1)

            # Mean of last 40 episodes
        print("Avg Reward is ==> {}".format(sum(avg_reward_list) / len(avg_reward_list)) + " with a 95 % confidence interval of +/- " + str(ci))
        for i in range(len(self.products)):
            print("achieved service level: " + str(np.mean(achieved_service_level[i])))


    def generate_products(self, n_periods, seed=None):
        first_index = 0
        last_index = 0
        generated_products = []
        for category in self.product_categories.keys():
            number_of_products = self.product_categories[category]
            last_index += number_of_products
            if category == "erratic":
                generated_products += generate_data.generate_seasonal_data_for_erratic_demand(self.products[first_index: last_index], n_periods, seed)
            elif category == "smooth":
                generated_products += generate_data.generate_seasonal_data_for_smooth_demand(self.products[first_index:last_index], n_periods, seed)
            else:
                generated_products += generate_data.generate_seasonal_data_for_intermittent_demand(self.products[first_index:last_index], n_periods, seed)
            first_index = last_index
        return generated_products
