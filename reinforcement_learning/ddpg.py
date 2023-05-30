import os
import random
import sys
import signal

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.models import load_model

from generate_data import generate_seasonal_data_based_on_products

# Define hyperparameters
gamma = 0.98  # discount factor
tau = 0.005  # target network update rate
actor_lr = 0.001  # learning rate of actor network
critic_lr = 0.002 # learning rate of critic network
batch_size = 100  # minibatch size
num_episodes = 2300
num_agents = 4  # number of agents
warm_up_steps = 1000


class Actor(models.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = layers.Dense(64, activation='selu')
        self.l2 = layers.Dense(64, activation='selu')
        self.l3 = layers.Dense(1, activation='sigmoid')
        self.max_action = max_action
        # Initialize weights between -3e-5 and 3-e5
        last_init = tf.random_uniform_initializer(minval=-0.00003, maxval=0.00003)

        # Actor will get observation of the agent
        # not the observation of other agents
        self.l1 = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")
        self.l2 = layers.Dropout(rate=0.5)
        self.l3 = layers.BatchNormalization()
        self.l4 = layers.Dense(256, activation="selu", kernel_initializer="lecun_normal")
        self.l5 = layers.Dropout(rate=0.5)
        self.l6 = layers.BatchNormalization()

        # Using tanh activation as action values for
        # for our environment lies between -1 to +1
        self.l7 = layers.Dense(1, activation="sigmoid", kernel_initializer=last_init)

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return abs(x)


class Critic(models.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.l1 = layers.Dense(32, activation='tanh')
        self.l2 = layers.Dense(32, activation='tanh')
        self.l3 = layers.Dense(1)
        self.flatten = layers.Flatten()

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # State as input, here this state is
        # observation of all the agents
        # hence this state will have information
        # of observation of all the agents
        self.l1 = layers.Dense(16, activation="selu", kernel_initializer="lecun_normal")
        self.l2 = layers.BatchNormalization()
        self.l3 = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")
        self.l4 = layers.BatchNormalization()
        # Action all the agents as input
        self.l9 = layers.Dense(32, activation="selu", kernel_initializer="lecun_normal")
        self.l10 = layers.BatchNormalization()

        self.l11 = layers.Dropout(rate=0.5)
        self.l12 = layers.BatchNormalization()
        self.l13 = layers.Dense(512, activation="selu", kernel_initializer="lecun_normal")
        self.l14 = layers.Dropout(rate=0.5)
        self.l15 = layers.BatchNormalization()

        self.outputs = layers.Dense(1)

    def call(self, state, actions):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        state_out = self.l4(x)
        a = self.l9(actions)
        action_out = self.l10(a)
        concat = layers.Concatenate()([state_out, action_out])
        out = self.l11(concat)
        out = self.l12(out)
        out = self.l13(out)
        out = self.l14(out)
        out = self.l15(out)
        out = self.outputs(out)

        return tf.reduce_mean(out, axis=1)


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

        next_actions = agents[0].actor_target(next_state)
        # next_actions = [tf.reshape(a, (-1,)) for a in next_actions]
        # next_actions = tf.stack(next_actions, axis=1)
        #
        #
        # # Compute the target Q value
        # next_actions = tf.expand_dims(next_actions, axis=-1)
        action = tf.expand_dims(action, axis=-1)
        target_Q = self.critic_target(next_state, next_actions)

        not_done = tf.cast(not_done, tf.float32)
        not_done = tf.reshape(not_done, (batch_size, 1))
        reward = reward.astype(np.float32)
        target_Q = tf.reshape(target_Q, [batch_size, 1])
        target_Q = reward + (not_done * self.discount * target_Q)

        # Compute critic loss
        with tf.GradientTape() as tape:
            current_Q = self.critic(state, action)
            critic_loss = tf.reduce_mean(tf.square(current_Q - target_Q))
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # gradients, _ = tf.clip_by_global_norm(gradients, 1)  # Apply gradient clipping
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        # Compute actor loss
        # Compute actor loss
        with tf.GradientTape() as tape:
            current_actions = agents[0].actor(state)
            # current_actions = [tf.reshape(a, (-1,)) for a in current_actions]
            # current_actions = tf.stack(current_actions, axis=1)
            # current_actions = tf.expand_dims(current_actions, axis=-1)
            actor_loss = -self.critic(state, current_actions)
            actor_loss = tf.reduce_mean(actor_loss)

            # Explicitly state that we want to watch the actor's variables
            tape.watch(self.actor.trainable_variables)

        # Optimize the actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, 0.1)  # Apply gradient clipping
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update the frozen target models
        self.update_target_networks()



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

    def update_target_networks(self, tau=None):
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
        return self.actor(state)[0][0]


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
        indices = np.random.choice(len(self.storage), size=batch_size)  # Sample indices with probabilities
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
        self.replay_buffer = ReplayBuffer()
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('Training interrupted. Saving models...')
        self.save_models()
        sys.exit(0)

    def train(self):
        # Main training loop
        # Warm-up phase
        state = self.env.reset()
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
        start_std_dev = 0.3
        noise_reduction = start_std_dev / num_episodes

        factor = 0

        for episode in range(num_episodes + 100):
            noise_std_dev = start_std_dev - episode * noise_reduction
            generate_seasonal_data_based_on_products(self.products, 500)
            done = False
            total_reward = 0
            state = self.env.reset()
            samples = []
            while not done:
                if episode > 100:
                    factor *= 0.95
                    if noise_std_dev <0.5:
                        noise_std_dev = 0.05
                    if factor < 0.05:
                        factor = 0
                # Select action according to policy
                if random.random() < factor:
                    actions = tf.random.uniform(shape=[4], minval=0, maxval=70)
                else:
                    random_number = random.uniform(-1, 1)
                    # actions = [agent.select_action(state[i])[0] + random_number * faktor for i, agent in enumerate(self.agents)]
                    actions = [np.clip(self.agents[0].select_action(state)[0], 0, 100)]
                    # Add Gaussian noise to the action
                    actions = actions + np.random.normal(0, noise_std_dev, size=len(self.products))

                    # Clip the action to make sure it's within the valid range
                    actions = np.clip(actions, 0, 100)

                # print("actions", actions)
                # Perform action and get reward
                next_state, reward, done, *_ = self.env.step(actions)
                total_reward += sum(reward)
                #
                # if (sum(reward)>-1150):
                #     print("yeaaah", reward)
                #     reward = [x + 1000 for x in reward]
                reward = total_reward
                # running_avg_reward = 0.99 * running_avg_reward + 0.01 * sum(reward) * 2
                # running_std_reward = np.sqrt(0.99 * running_std_reward ** 2 + 0.01 * (sum(reward) * 2 - running_avg_reward) ** 2)
                # reward = [-abs((reward - running_avg_reward) / running_std_reward) for reward in reward]

                if (episode > 100):
                    # Store experience in replay buffer
                    self.replay_buffer.add((state, actions, next_state, reward, done))

                # Move to next state
                state = next_state

                # Train agent
                if episode > 200:
                    if episode % 300 == 0:
                        for agent_num in range(len(self.agents)):
                            self.agents[agent_num].learn(self.replay_buffer, self.agents, agent_num, batch_size=len(self.replay_buffer.storage))
                    else:
                        for agent_num in range(len(self.agents[:1])):
                            self.agents[agent_num].learn(self.replay_buffer, self.agents, agent_num, batch_size=batch_size)
            if episode > 200 or episode % 10 == 0:
                print(actions)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        self.save_models()

    def test(self, start_time_period=208):
        # Want to print the environment, costs and actions:
        self.env.verbose = True
        self.env.time_period = start_time_period
        generate_seasonal_data_based_on_products(self.products, 500)
        self.actors = []
        # Load the actor networks
        for i, agent in enumerate(self.agents):
            loaded_model = load_model(os.path.join('models', f'actor_model_{i}'))
            agent.actor = loaded_model
            # agent.actor_target(self.load_models(os.path.join('models', f'actor_target_model_{i}.h5')))
            # agent.critic(self.load_models(os.path.join('models', f'critic_model_{i}.h5')))
            # agent.critic_target(self.load_models(os.path.join('models', f'critic_target_model_{i}.h5')))

        done = False
        total_costs = 0

        self.env.time_period = start_time_period

        sum_rewards = 0

        for episode in range(num_episodes):
            generate_seasonal_data_based_on_products(self.products, 500)
            done = False
            total_reward = 0
            state = self.env.reset()
            samples = []
            count = 0
            while not done:
                print(f"Inventory_level at start of period {self.env.current_period}: {self.env.inventory_levels}")

                actions = [np.clip(agent.select_action(state[i])[0], 0, 100) for i, agent in enumerate(self.agents)]
                count += 1
                next_state, reward, done, *_ = self.env.step(actions)
                total_reward += sum(reward)
                self.replay_buffer.add((state, actions, next_state, reward, done))
                print(f"Action for time period {self.env.current_period}: {actions}")

                # Move to next state
                state = next_state

                if done:
                    sum_rewards += total_reward
            print("COUNTTT", count)
        print("Average total costs: ", sum_rewards / num_episodes)

    def save_models(self):
        save_dir = 'models'
        os.makedirs(save_dir, exist_ok=True)
        print('Saving models...')
        for i, agent in enumerate(self.agents):
            agent.actor.save(os.path.join(save_dir, f'actor_model_{i}'), save_format='tf')



