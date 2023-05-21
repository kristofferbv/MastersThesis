import random

import numpy as np
from gurobipy import Model
from keras import Sequential, Input, optimizers
from keras.layers import Dense, Flatten
import tensorflow as tf

from config_utils import load_config
import keras


def get_greedy_action_2(prob_distributions):
    # Assuming `prob_distributions` is the nested list of probability distributions
    actions_list = []
    for product_probs in prob_distributions[0]:
        action = np.argmax(product_probs)  # Select action with highest probability
        actions_list.append(action)
    return actions_list


# For each product, we randomly select an action based on the provided probability distribution.
def get_stochastic_action_2(prob_distributions):
    # Assuming `prob_distributions` is the nested list of probability distributions
    actions_list = []
    for product_probs in prob_distributions[0]:
        actions = np.arange(len(product_probs))  # Generate an array of action indices
        action = np.random.choice(actions, p=product_probs)  # Sample an action
        actions_list.append(action)
    return actions_list


# For each product, we randomly select an action based on the provided probability distribution.
def get_greedy_action(prob_distributions):
    # `prob_distributions` is now a 1D array of joint action probabilities
    joint_action = np.argmax(prob_distributions[0])  # Select joint action with highest probability
    return joint_action


def get_stochastic_action(prob_distributions):
    # `prob_distributions` is now a 1D array of joint action probabilities
    actions = np.arange(len(prob_distributions[0]))  # Generate an array of joint action indices
    try:
        if np.random.random() < 0.2:
            return random.choice(actions)
        else:
            joint_action = np.random.choice(actions, p=prob_distributions[0])  # Sample a joint action
    except:
        print("probability distribution: ", prob_distributions)
    return joint_action


def unflatten_action(joint_action, num_actions=6, num_products=6):
    actions_list = []
    for _ in range(num_products):
        actions_list.append(joint_action % num_actions)
        joint_action //= num_actions
    return actions_list


class Actor:
    def __init__(self, input_shape, output_size, name="default_name", model_path=None) -> None:
        self.name = name
        self.input_shape = input_shape
        self.output_size = output_size
        config = load_config("config.yml")
        actor_config = config["actor"]
        self.learning_rate = actor_config["learning_rate"]
        self.output_activation_function = actor_config["output_activation_function"]
        self.batch_size = actor_config["batch_size"]
        self.epochs = actor_config["epochs"]
        self.optimizer = actor_config["optimizer"]
        self.loss_function = actor_config["loss_function"]
        self.epsilon = actor_config["epsilon"]
        self.epsilon_decay = actor_config["epsilon_decay"]
        self.epsilon_min_value = actor_config["epsilon_min_value"]
        self.stochastic = actor_config["stochastic"]
        self.episodes_epsilon_one = actor_config["episodes_epsilon_one"]

        # Layers
        self.dense_layers = actor_config["layers"]
        self.dense_activation_functions = actor_config["activation_functions"]

        if model_path is None:
            self.model = self.create_network(self.optimizer, self.loss_function)
            for i, layer in enumerate(self.model.layers):
                print(f"Layer {i + 1} ({layer.__class__.__name__}):  {layer.input_shape} + {layer.output_shape}")

        else:
            self.model = keras.models.load_model(model_path)

    def get_stochastic_action(self, prob_distributions):
        # `prob_distributions` is now a 1D array of joint action probabilities
        actions = np.arange(len(prob_distributions[0]))  # Generate an array of joint action indices
        try:
            if np.random.random() < self.epsilon:
                joint_action = random.choice(actions)
            else:
                joint_action = np.random.choice(actions, p=prob_distributions[0])  # Sample a joint action
        except:
            print("probability distribution: ", prob_distributions)
        return joint_action

    def reduce_exploration_rate(self, episode):
        if episode > 100 and self.epsilon != self.epsilon_min_value:
            self.epsilon = self.epsilon * self.epsilon_decay
            print("EPSILON ", self.epsilon)
        if self.epsilon < self.epsilon_min_value:
            self.epsilon = self.epsilon_min_value

    def create_network(self, optimizer, loss_function):
        optimizer = eval("keras.optimizers." + optimizer)
        self.optimizer = optimizer(learning_rate=self.learning_rate)
        loss_function = eval("keras.losses." + loss_function)

        model = Sequential()
        print("INPUS SHA", self.input_shape)
        model.add(Input(shape=self.input_shape))
        model.add(Flatten())

        for i in range(len(self.dense_layers)):
            model.add(Dense(int(self.dense_layers[i]), activation=self.dense_activation_functions[i]))

        # Change the output size to represent all possible joint actions.
        model.add(Dense(self.output_size, activation=self.output_activation_function))

        model.compile(optimizer=self.optimizer,
                      loss=loss_function)

        return model


    def predict(self, state):
        """
        Predict the probability distribution over actions given a state.
        """
        # Reshape the state to have an extra dimension because the model expects a batch of states
        state = np.expand_dims(state, axis=0)

        # Get the probability distribution over actions
        action_prob = self.model.predict(state, verbose=0)
        # Return the action probabilities
        # weights = self.model.get_weights()
        # for i, layer_weights in enumerate(weights):
        #     print(f"Layer {i} weights:")
        #     print(layer_weights)
        return action_prob
