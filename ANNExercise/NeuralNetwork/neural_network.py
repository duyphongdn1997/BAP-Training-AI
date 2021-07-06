import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    """
    This is Neural Network class
    """
    def __init__(self, seed: int = 1):
        """
        This method used to initialize the init value
        """
        self.output_values = []
        self.summary = []
        self.layer_names = []
        self.activation_names = []
        self.activation_functions = []
        self.weights = []
        self.bias = []
        self.output_shape = []
        self.n_units = []
        self.params = []
        self.dw = []
        self.db = []
        self.seed = seed
        self.number_of_layers: int = 0
        self.activations = []
        np.random.seed(self.seed)

    @staticmethod
    def forward(x, weight):
        """
        This method used to compute output of the only current layer.
        :param x: Input of the layer
        :param weight: the weights used to compute output z of current layer.
        :return: the output z of the layer
        """
        return weight @ x

    @staticmethod
    def _init_weights(l_in, l_out):
        """
        This method used to initialize random weights
        param: l_in: the number of neuron in the input layer
        param: l_out: the number of neuron in the output layer
        return: initial weights of the current layer.
        """
        epsilon = 0.12
        W = np.random.rand(l_out, l_in) * 2 * epsilon - epsilon
        return W

    def __feed_forward(self, x):
        """
        This method used to compute feed forward
        :param x: the input variables
        :return: the predict of the x using the current neuron network.
        """
        a = 0
        for layer_index in range(self.number_of_layers - 1):
            z = self.forward(x, self.weights[layer_index]) + self.bias[layer_index]
            self.output_values.append(z)
            a = self.activation_functions[layer_index](z)
            self.activations.append(a)
        return a

    def __backpropagation(self, y_true: np.ndarray, y_predict: np.ndarray):
        """
        This method used to compute and update the params for each epoch
        :param y_true: The target variables
        :param y_predict: the result value after feed forward the neural network
        :return: new param of the neural network
        """
        pass

    def add(self, layer_name: str = "", n_unit: int = 1, activation: str = None):
        """
        This method used to adding a layer instance on top of the layer stack.
        :param layer_name: The name of layer, could be "input", "dense", "output"
        :param n_unit: The number of unit in current layer
        :param activation: The name of activation function apply to current layer
        """
        self.number_of_layers += 1
        self.layer_names.append(layer_name)
        self.n_units.append(n_unit)
        if self.number_of_layers > 1:
            # Create init weight and bias
            self.weights.append(self._init_weights(self.n_units[self.number_of_layers - 1],
                                                   self.n_units[self.number_of_layers]))
            self.params.append(self.n_units[self.number_of_layers - 1] * self.n_units[self.number_of_layers])
            self.bias.append(np.zeros((n_unit, 1)))
            if activation is None:
                self.activation_names.append("None")
                self.activation_functions.append(None)
            elif activation == "sigmoid":
                self.activation_names.append("sigmoid")
                self.activation_functions.append(self.__sigmoid)
            elif activation == "ReLU":
                self.activation_names.append("ReLU")
                self.activation_functions.append(self.__relu)
            elif activation == "ELU":
                self.activation_names.append("ELU")
                self.activation_functions.append(self.__elu)
            elif activation == "tanh":
                self.activation_names.append("tanh")
                self.activation_functions.append(self.__tanh)
            elif activation == "LeakyReLU":
                self.activation_names.append("LeakyReLU")
                self.activation_functions.append(self.__leaky_relu)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            epochs: int, batch_size: int, learning_rate: float = 0.001):
        """
        This method used to train the model with x_train and y_train dataset.
        :param x_train: input variables
        :param y_train: target variables
        :param epochs: the number of iterations to loop the training section.
        :param batch_size: the size of input variables feed to neural network for training
        :param learning_rate: the learning rate to update parameter.
        :return: the weights after training
        """
        pass

    def summary(self):
        """
        This method used to print a string summary of the neural network.
        :return:
        """
        pass

    def pop(self):
        """
        This method used to remove the last layer in the model.
        """
        self.number_of_layers -= 1
        self.activation_names.pop(-1)
        self.params.pop(-1)
        self.n_units.pop(-1)
        self.layer_names.pop(-1)

    def __gradient_descent(self):
        """
        This method used to compute the gradient descent.
        :return:
        """
        pass

    @staticmethod
    def __sigmoid(z):
        """
        This method used to compute sigmoid function
        :param: z: input of sigmoid function
        :return: result after use sigmoid function to z.
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __add_bias(x: np.ndarray):
        """
        This method used to computes the cost of using theta as the parameter for regularized linear regression
        to fit the data in X and y.
        :param: X: Input variables
        :return: cost of using theta as the parameter for regularized linear regression to fit the data in X and y.
        """
        m = len(x)
        bias = np.ones(m)
        X = np.vstack((bias, x.T)).T
        return X

    @staticmethod
    def __relu(z):
        """
        This method used to compute the ReLU function
        :param z: input variables
        :return: Output of ReLU function
        """
        return z if z > 0 else 0

    @staticmethod
    def __leaky_relu(z, alpha: float = 0.01):
        """
        This method used to compute the leaky ReLU function
        :param z: input variables
        :param alpha: input variables
        :return: Output of Leaky ReLU function
        """
        return z if z > 0 else alpha * z

    @staticmethod
    def __elu(z, alpha: float = 1):
        """
        This method used to compute the ELU function
        :param z: input variables
        :param alpha: input variables
        :return: Output of ELU function
        """
        return z if z > 0 else alpha * (np.exp(z) - 1)

    @staticmethod
    def __tanh(z):
        """
        This method used to compute the tanh function
        :param z: input variables
        :return: Output of tanh function
        """
        return np.tanh(z)

    def predict(self, x_test: np.ndarray):
        """
        This method used to predict x through the neural network model.
        :param x_test: the input variables
        :return: the predict variables
        """
        pass

    def accuracy_score(self, y_val: np.ndarray, y_predict: np.ndarray):
        """
        This method used to compute the accuracy score of the y_predict
        :param y_val: the actual variables
        :param y_predict: the predict value we want to compute score.
        :return: the accuracy score.
        """
        pass
