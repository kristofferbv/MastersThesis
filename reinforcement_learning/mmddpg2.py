import json

import numpy as np
import tensorflow as tf
from keras import layers, models
import keras
from keras.layers import Dense



class ReplayBuffer():
    def __init__(self, env, buffer_capacity=200000, batch_size=64, min_size_buffer=100):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.n_agents = env.n
        self.list_actors_dimension = [env.observation_space[index].shape[0] for index in range(self.n_agents)]
        self.critic_dimension = sum(self.list_actors_dimension)
        self.list_actor_n_actions = [env.action_space[index].n for index in range(self.n_agents)]

        self.states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.rewards = np.zeros((self.buffer_capacity, self.n_agents))
        self.next_states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.dones = np.zeros((self.buffer_capacity, self.n_agents), dtype=bool)

        self.list_actors_states = []
        self.list_actors_next_states = []
        self.list_actors_actions = []

        for n in range(self.n_agents):
            self.list_actors_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
            self.list_actors_next_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
            self.list_actors_actions.append(np.zeros((self.buffer_capacity, self.list_actor_n_actions[n])))

    def __len__(self):
        return self.buffer_counter

    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer

    def update_n_games(self):
        self.n_games += 1

    def add_record(self, actor_states, actor_next_states, actions, state, next_state, reward, done):

        index = self.buffer_counter % self.buffer_capacity

        for agent_index in range(self.n_agents):
            self.list_actors_states[agent_index][index] = actor_states[agent_index]
            self.list_actors_next_states[agent_index][index] = actor_next_states[agent_index]
            self.list_actors_actions[agent_index][index] = actions[agent_index]

        self.states[index] = state
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done

        self.buffer_counter += 1

    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records,
        # if the cunter is higher we don't access the record using the counter
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)

        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Take indices
        state = self.states[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]

        actors_state = [self.list_actors_states[index][batch_index] for index in range(self.n_agents)]
        actors_next_state = [self.list_actors_next_states[index][batch_index] for index in range(self.n_agents)]
        actors_action = [self.list_actors_actions[index][batch_index] for index in range(self.n_agents)]

        return state, reward, next_state, done, actors_state, actors_next_state, actors_action

    def save(self, folder_path):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        np.save(folder_path + '/states.npy', self.states)
        np.save(folder_path + '/rewards.npy', self.rewards)
        np.save(folder_path + '/next_states.npy', self.next_states)
        np.save(folder_path + '/dones.npy', self.dones)

        for index in range(self.n_agents):
            np.save(folder_path + '/states_actor_{}.npy'.format(index), self.list_actors_states[index])
            np.save(folder_path + '/next_states_actor_{}.npy'.format(index), self.list_actors_next_states[index])
            np.save(folder_path + '/actions_actor_{}.npy'.format(index), self.list_actors_actions[index])

        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}

        with open(folder_path + '/dict_info.json', 'w') as f:
            json.dump(dict_info, f)

    def load(self, folder_path):
        self.states = np.load(folder_path + '/states.npy')
        self.rewards = np.load(folder_path + '/rewards.npy')
        self.next_states = np.load(folder_path + '/next_states.npy')
        self.dones = np.load(folder_path + '/dones.npy')

        self.list_actors_states = [np.load(folder_path + '/states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_next_states = [np.load(folder_path + '/next_states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_actions = [np.load(folder_path + '/actions_actor_{}.npy'.format(index)) for index in range(self.n_agents)]

        with open(folder_path + '/dict_info.json', 'r') as f:
            dict_info = json.load(f)
        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]

class Critic(keras.Model):
    def __init__(self, name, hidden_0=64, hidden_1=64):
        super(Critic, self).__init__()

        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)

    def call(self, state, actors_actions):
        state_action_value = self.dense_0(tf.concat([state, actors_actions], axis=1))  # multiple actions
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)

        return q_value

class Actor(keras.Model):
    def __init__(self, name, actions_dim, hidden_0=64, hidden_1=64):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.policy = Dense(self.actions_dim, activation='sigmoid')  # we want something beetween zero and one

    def call(self, state):
        x = self.dense_0(state)
        policy = self.dense_1(x)
        policy = self.policy(policy)
        return policy

class Agent:
    def __init__(self, env, n_agent, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.05):

        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_dims = env.observation_space[n_agent].shape[0]
        self.n_actions = env.action_space[n_agent].n

        self.agent_name = "agent_number_{}".format(n_agent)

        self.actor = Actor("actor_" + self.agent_name, self.n_actions)
        self.critic = Critic("critic_" + self.agent_name)
        self.target_actor = Actor("target_actor_" + self.agent_name, self.n_actions)
        self.target_critic = Critic("critic_" + self.agent_name)

        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))

        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)

    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights

        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.target_critic.set_weights(target_critic_weights)

    def get_actions(self, actor_states):
        noise = tf.random.uniform(shape=[self.n_actions])
        actions = self.actor(actor_states)
        actions = actions + noise

        return actions.numpy()[0]

    def save(self, path_save):
        self.actor.save_weights(f"{path_save}/{self.actor.net_name}.h5")
        self.target_actor.save_weights(f"{path_save}/{self.target_actor.net_name}.h5")
        self.critic.save_weights(f"{path_save}/{self.critic.net_name}.h5")
        self.target_critic.save_weights(f"{path_save}/{self.target_critic.net_name}.h5")

    def load(self, path_load):
        self.actor.load_weights(f"{path_load}/{self.actor.net_name}.h5")
        self.target_actor.load_weights(f"{path_load}/{self.target_actor.net_name}.h5")
        self.critic.load_weights(f"{path_load}/{self.critic.net_name}.h5")
        self.target_critic.load_weights(f"{path_load}/{self.target_critic.net_name}.h5")

class SuperAgent:
    def __init__(self, env, path_save=PATH_SAVE_MODEL, path_load=PATH_LOAD_FOLDER):
        self.path_save = path_save
        self.path_load = path_load
        self.replay_buffer = ReplayBuffer(env)
        self.n_agents = len(env.agents)
        self.agents = [Agent(env, agent) for agent in range(self.n_agents)]

    def get_actions(self, agents_states):
        list_actions = [self.agents[index].get_actions(agents_states[index]) for index in range(self.n_agents)]
        return list_actions

    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        full_path = f"{self.path_save}/save_agent_{date_now}"
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

        for agent in self.agents:
            agent.save(full_path)

        self.replay_buffer.save(full_path)

    def load(self):
        full_path = self.path_load
        for agent in self.agents:
            agent.load(full_path)

        self.replay_buffer.load(full_path)

    def train(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, reward, next_state, done, actors_state, actors_next_state, actors_action = self.replay_buffer.get_minibatch()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_state, dtype=tf.float32)

        actors_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_state]
        actors_next_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_next_state]
        actors_actions = [tf.convert_to_tensor(s, dtype=tf.float32) for s in actors_action]

        with tf.GradientTape(persistent=True) as tape:
            target_actions = [self.agents[index].target_actor(actors_next_states[index]) for index in range(self.n_agents)]
            policy_actions = [self.agents[index].actor(actors_states[index]) for index in range(self.n_agents)]

            concat_target_actions = tf.concat(target_actions, axis=1)
            concat_policy_actions = tf.concat(policy_actions, axis=1)
            concat_actors_action = tf.concat(actors_actions, axis=1)

            target_critic_values = [tf.squeeze(self.agents[index].target_critic(next_states, concat_target_actions), 1) for index in range(self.n_agents)]
            critic_values = [tf.squeeze(self.agents[index].critic(states, concat_actors_action), 1) for index in range(self.n_agents)]
            targets = [rewards[:, index] + self.agents[index].gamma * target_critic_values[index] * (1 - done[:, index]) for index in range(self.n_agents)]
            critic_losses = [tf.keras.losses.MSE(targets[index], critic_values[index]) for index in range(self.n_agents)]

            actor_losses = [-self.agents[index].critic(states, concat_policy_actions) for index in range(self.n_agents)]
            actor_losses = [tf.math.reduce_mean(actor_losses[index]) for index in range(self.n_agents)]

        critic_gradients = [tape.gradient(critic_losses[index], self.agents[index].critic.trainable_variables) for index in range(self.n_agents)]
        actor_gradients = [tape.gradient(actor_losses[index], self.agents[index].actor.trainable_variables) for index in range(self.n_agents)]

        for index in range(self.n_agents):
            self.agents[index].critic.optimizer.apply_gradients(zip(critic_gradients[index], self.agents[index].critic.trainable_variables))
            self.agents[index].actor.optimizer.apply_gradients(zip(actor_gradients[index], self.agents[index].actor.trainable_variables))
            self.agents[index].update_target_networks(self.agents[index].tau)

env = make_env(ENV_NAME)
super_agent = SuperAgent(env)

scores = []

if PATH_LOAD_FOLDER is not None:
    print("loading weights")
    actors_state = env.reset()
    actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
    [super_agent.agents[index].target_actor(actors_state[index][None, :]) for index in range(super_agent.n_agents)]
    state = np.concatenate(actors_state)
    actors_action = np.concatenate(actors_action)
    [super_agent.agents[index].critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    [super_agent.agents[index].target_critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    super_agent.load()

    print(super_agent.replay_buffer.buffer_counter)
    print(super_agent.replay_buffer.n_games)



for n_game in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    actors_state = env.reset()
    done = [False for index in range(super_agent.n_agents)]
    score = 0
    step = 0

    if (super_agent.replay_buffer.n_games + 1) > 5000:
        MAX_STEPS = int((super_agent.replay_buffer.n_games + 1) / 200)

    while not any(done):
        actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
        actors_next_state, reward, done, info = env.step(actors_action)

        state = np.concatenate(actors_state)
        next_state = np.concatenate(actors_next_state)

        super_agent.replay_buffer.add_record(actors_state, actors_next_state, actors_action, state, next_state, reward, done)

        actors_state = actors_next_state

        score += sum(reward)
        step += 1
        if step >= MAX_STEPS:
            break

    if super_agent.replay_buffer.check_buffer_size():
        super_agent.train()

    super_agent.replay_buffer.update_n_games()

    scores.append(score)

    wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter,
               "Average reward": round(np.mean(scores[-10:]), 2), \
               "Time taken": round(time.time() - start_time, 2), 'Max steps': MAX_STEPS})

    if (n_game + 1) % EVALUATION_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
        actors_state = env.reset()
        done = [False for index in range(super_agent.n_agents)]
        score = 0
        step = 0
        while not any(done):
            actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
            actors_next_state, reward, done, info = env.step(actors_action)
            state = np.concatenate(actors_state)
            next_state = np.concatenate(actors_next_state)
            actors_state = actors_next_state
            score += sum(reward)
            step += 1
            if step >= MAX_STEPS:
                break
        wandb.log({'Game number': super_agent.replay_buffer.n_games,
                   '# Episodes': super_agent.replay_buffer.buffer_counter,
                   'Evaluation score': score, 'Max steps': MAX_STEPS})

    if (n_game + 1) % SAVE_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
        print("saving weights and replay buffer...")
        super_agent.save()
        print("saved")