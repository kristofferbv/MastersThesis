from MIP.config_utils import load_config


class Actor:
    def __init__(self):
        def __init__(self, possible_moves, input_size, output_size, name="default_name", model_path=None) -> None:
            self.name = name
            self.output_size = output_size
            self.all_possible_moves = possible_moves
            self.input_size = input_size
            self.epsilon_reduced_count = 0

            config = load_config("config.yml")
            self.learning_rate = config["neural_network"]["learning_rate"]
            self.output_activation_function = config["neural_network"]["output_activation_function"]
            self.batch_size = config["neural_network"]["batch_size"]
            self.epochs = config["neural_network"]["epochs"]
            self.optimizer = config["neural_network"]["optimizer"]
            self.loss_function = config["neural_network"]["loss_function"]
            self.epsilon = config["neural_network"]["epsilon"]
            self.epsilon_decay = config["neural_network"]["epsilon_decay"]
            self.epsilon_min_value = config["neural_network"]["epsilon_min_value"]
            self.stochastic = config["neural_network"]["stochastic"]
            self.episodes_epsilon_one = config["neural_network"]["episodes_epsilon_one"]

            # CNN
            self.cnn = config["neural_network"]["cnn_layers"]["cnn"]
            self.cnn_activation_function = config["neural_network"]["cnn_layers"]["activation_function"]
            self.filters = config["neural_network"]["cnn_layers"]["filters"]
            self.kernel_size = config["neural_network"]["cnn_layers"]["kernel_size"]
            self.pool_size = config["neural_network"]["cnn_layers"]["pool_size"]

            # Dense Layers
            self.dense_layers = config["neural_network"]["dense_layers"]["layers"]
            self.dense_activation_functions = config["neural_network"]["dense_layers"]["activation_functions"]

            if model_path is None:
                self.model = self._init_network(self.optimizer, self.loss_function)
            else:
                self.model = keras.models.load_model(model_path)
