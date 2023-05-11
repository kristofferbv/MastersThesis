import numpy as np

from config_utils import load_config
import keras
from keras import Sequential, Input
from keras.layers import Dense, Flatten


class Critic:
    def __init__(self, input_shape, name="default_name", model_path=None) -> None:
        self.name = name
        self.input_shape = input_shape
        config = load_config("config.yml")
        critic_config = config["critic"]
        self.learning_rate = critic_config["learning_rate"]
        self.output_activation_function = critic_config["output_activation_function"]
        self.batch_size = critic_config["batch_size"]
        self.epochs = critic_config["epochs"]
        self.optimizer = critic_config["optimizer"]
        self.loss_function = critic_config["loss_function"]
        self.epsilon = critic_config["epsilon"]
        self.epsilon_decay = critic_config["epsilon_decay"]
        self.epsilon_min_value = critic_config["epsilon_min_value"]
        self.stochastic = critic_config["stochastic"]
        self.episodes_epsilon_one = critic_config["episodes_epsilon_one"]

        # Layers
        self.dense_layers = critic_config["layers"]
        self.dense_activation_functions = critic_config["activation_functions"]

        if model_path is None:
            self.model = self.create_network(self.optimizer, self.loss_function)
        else:
            self.model = keras.models.load_model(model_path)

    def create_network(self, optimizer, loss_function):
        optimizer = eval("keras.optimizers." + optimizer)
        self.optimizer = optimizer(learning_rate=self.learning_rate)
        loss_function = eval("keras.losses." + loss_function)

        model = Sequential()
        model.add(Input(shape=self.input_shape))
        # Need to flatten the architecture (I think)
        model.add(Flatten())

        # model.add(Input(shape=(self.input_size,)))

        for i in range(len(self.dense_layers)):
            model.add(Dense(int(self.dense_layers[i]), activation=self.dense_activation_functions[i]))
        model.add(Dense(1))
        model.compile(optimizer=self.optimizer,
                      loss=loss_function, metrics=[keras.metrics.categorical_accuracy])

        return model

    def predict(self, state):
        """
        Predict the value function given a state.
        """
        # Reshape the state to have an extra dimension because the model expects a batch of states
        state = np.expand_dims(state, axis=0)

        # Get the value function
        value = self.model.predict(state, verbose=0)

        # Return the value
        return value

    def fit(self, states, targets, verbose=0):
        """
        Train the critic network given the states and targets.
        """
        # Reshape the states and targets to match the expected input shapes
        states = np.array(states)
        states = states.reshape(1, 6, 6)
        targets = np.array(targets)
        targets = targets.reshape(1, 1)

        # Train the critic network
        self.model.fit(states, targets, verbose=0, epochs=100)

