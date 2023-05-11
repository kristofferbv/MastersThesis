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

    def train_a2c(self, n_episodes = None):
        if n_episodes is None:
            n_episodes = self.n_episodes
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action_prob = self.actor.predict(state)
                print(action_prob)
                # print(action_prob)
                # getting action based on probability distribution
                action = get_stochastic_action(action_prob)
                individual_actions = unflatten_action(action)
                print(individual_actions)
                # Use this for stochastic action if probability distribution is an 1D array:
                # action = np.random.choice(len(action_prob[0]), p=action_prob[0])

                # print("actions")
                # print(action)
                next_state, reward, done, _ = self.env.step(individual_actions)
                total_reward += reward
                target = reward + (1 - done) * self.discount_rate * self.critic.predict(next_state)
                td_error = target - self.critic.predict(state)
                # print(td_error)

                # Train the Critic
                # print("PREDICTION", self.critic.predict(state))

                self.critic.fit(state, target, verbose=0)

                # Train the Actor
                with tf.GradientTape() as tape:
                    state_batch = np.expand_dims(state, axis=0)
                    action_prob = self.actor.model(state_batch)
                    log_prob = tf.math.log(tf.reduce_sum(action_prob * action, axis=1))
                    actor_loss = -tf.reduce_mean(td_error * log_prob)
                actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))

                state = next_state
            print(f'Epoch {episode + 1}/{n_episodes}: Total Reward: {total_reward}')

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