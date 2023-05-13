import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
from config_utils import load_config
from reinforcement_learning.actor import get_stochastic_action, unflatten_action

"""
state_t = s 
action = a
state_t+1 = s'
reward_t+1 = r


V(s) = r + discount_factor * V(s')

s' = last_state, v(s') = reward

"""

class A2CAgent:
    def __init__(self, actor, critic, env):
        self.actor = actor
        self.critic = critic
        config = load_config("config.yml")
        self.discount_rate = config["a2c"]["discount_rate"]
        self.n_episodes = config["a2c"]["n_episodes"]
        self.actor_optimizer = actor.optimizer
        self.critic_optimizer = critic.optimizer
        self.env = env

    def train_a2c(self, n_episodes = None, verbose = False):
        if n_episodes is None:
            n_episodes = self.n_episodes
        # Initialize running average and standard deviation of rewards
        running_avg_reward = 0
        running_std_reward = 1  # Initialize to 1 to avoid division by zero issues
        batch_states, batch_actions, batch_td_errors = [], [], []
        for episode in range(n_episodes):
            states = self.env.reset()
            done = False
            total_reward = 0
            td_errors = []

            while not done:
                actions = []
                for i, state in enumerate(states):
                    action_prob = self.actor.predict(state)
                    action = get_stochastic_action(action_prob)
                    actions.append(action)
                next_states, rewards, done, _ = self.env.step(actions)
                total_reward += sum(rewards)

                running_avg_reward = 0.99 * running_avg_reward + 0.01 * sum(rewards)
                running_std_reward = np.sqrt(0.99 * running_std_reward ** 2 + 0.01 * (sum(rewards) - running_avg_reward) ** 2)
                rewards = [(reward - running_avg_reward) / running_std_reward for reward in rewards]

                target = sum(rewards) + (1 - done) * self.discount_rate * self.critic.predict(next_states)
                td_error = target - self.critic.predict(states)
                td_errors.append(td_error)
                batch_states.extend(np.expand_dims(states, axis=0))
                batch_actions.extend(actions)
                batch_td_errors.extend(td_error)
                if verbose:
                    print("actions", actions)
                    print("total reward: ", total_reward)
                    print("prdiction next", self.critic.predict(next_states))
                    print("state: ", states)
                    print("next state: ", next_states)
                    print("td error: ", td_error)
                states = next_states

            # if len(batch_states) >= 13:
            # TODO! fix this
            batch_states = np.array(batch_states)
            with tf.GradientTape() as tape:
                state_values = self.critic.model(batch_states)
                critic_loss = tf.reduce_mean(tf.square(np.array(batch_td_errors) - state_values))
            critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.model.trainable_variables))

            for state, action, td_error in zip(batch_states, batch_actions, batch_td_errors):
                with tf.GradientTape() as tape:
                    action_prob = self.actor.model(state)
                    epsilon = 1e-8  # small constant to avoid zero values
                    log_prob = tf.math.log(tf.reduce_sum(action_prob * action, axis=1) + epsilon)
                    actor_loss = -tf.reduce_mean(td_error * log_prob)
                actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
                actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, 1.0)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))

            # with tf.GradientTape() as tape:
            #     state_values = self.critic.model(np.expand_dims(np.array(states), axis=0))
            #     critic_loss = tf.reduce_mean(tf.square(target - state_values))
            # critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)
            # self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.model.trainable_variables))
            #
            # for i, (state, action, td_error) in enumerate(zip(states, actions, td_errors)):
            #     with tf.GradientTape() as tape:
            #         state_batch = np.expand_dims(state, axis=0)
            #         action_prob = self.actor.model(state_batch)
            #         epsilon = 1e-8  # small constant to avoid zero values
            #         log_prob = tf.math.log(tf.reduce_sum(action_prob * action, axis=1) + epsilon)
            #         actor_loss = -tf.reduce_mean(td_error * log_prob)
            #     actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
            #     # Apply gradient clipping
            #     self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))
            # Clear the batch
            batch_states, batch_actions, batch_td_errors = [], [], []




            print(f'Epoch {episode + 1}/{n_episodes}: Total Reward: {total_reward}')
            print("sum td errors: ", sum(abs(x) for x in td_errors))
            print("std dev td errors: ", np.std(td_errors))

    def evaluate_a2c(a2c_model, env, n_episodes):
        total_rewards = []

        for episode in range(n_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action_prob = a2c_model.actor.predict(np.expand_dims(state, axis=0))
                action = np.argmax(action_prob[0])

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f'Average reward over {n_episodes} episodes: {avg_reward}')
        return avg_reward