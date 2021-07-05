import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork(object):
    """

    """
    def __init__(self, X: np.ndarray, y: np.ndarray, epochs: int,
                 batch_size: int, learning_rate: float = 0.01,
                 lambda_: float = 0.001):
        """

        """
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_

    def _init_weights(self, l0, l1):
        """
        This method used to create the initial weights for neural network.
        :param l0:
        :param l1:
        :return:
        """
        pass

    def feed_forward(self, X, parameter):
        """
        This method used to compute
        :param X:
        :param parameter:
        :return:
        """

    def add(self, layer_name: str = "", n_unit: int = 1, activation: str = None):
        """
        This method used to adding a layer instance on top of the layer stack.
        :param layer_name: The name of layer, could be "input", "dense", "output"
        :param n_unit: The number of unit in current layer
        :param activation: The name of activation function apply to current layer
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
        pass

    def __gradient_descent(self):
        """
        This method used to compute the gradient descent.
        :return:
        """
        pass

    # def __compute_cost(self, y: np.ndarray):
