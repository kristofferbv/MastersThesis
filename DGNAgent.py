from collections import deque
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.create_dqn_model(state_size, action_size)

    def create_dqn_model(self, state_size, action_size):
        model = Sequential()
        model.add(Dense(128, input_dim=state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_dqn_agent(self, agent, env, episodes, time_steps, batch_size):
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])

            for t in range(time_steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"episode: {e}/{episodes}, score: {t}, e: {agent.epsilon:.2}")
                    break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
