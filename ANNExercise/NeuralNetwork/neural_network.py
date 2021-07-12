import matplotlib.pyplot as plt
# from tqdm import tqdm
import numpy as np

from ANNExercise.NeuralNetwork.activation import *
from ANNExercise.NeuralNetwork.util import *


class NeuralNetwork(object):
    """
    This is Neural Network class
    """

    def __init__(self):
        """
        This method used to initialize the init value
        """
        self.backward_functions = []
        self.dz = []
        self.output_values = []
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
        self.history = []
        self.number_of_layers: int = 0
        self.activations = []

    @staticmethod
    def _init_weights(l_out, l_in):
        """
        This method used to initialize random weights
        param: l_in: the number of neuron in the input layer
        param: l_out: the number of neuron in the output layer
        return: initial weights of the current layer.
        """
        epsilon = np.sqrt(1.0 / l_out)
        W = np.random.randn(l_out, l_in) * epsilon
        return W

    def __feed_forward(self, x):
        """
        This method used to compute feed forward
        :param x: the input variables
        :return: the predict of the x using the current neuron network.
        """
        # self.output_values.append(x)
        # self.activations.append(x)
        self.activations.append(x)
        # z = (self.weights[0].T @ x) + self.bias[0]
        z = x.dot(self.weights[0]) + self.bias[0]
        self.output_values.append(z)
        a = self.activation_functions[0](z)
        self.activations.append(a)
        for layer_index in range(1, self.number_of_layers - 1):
            # z = self.weights[layer_index] @ self.activations[layer_index]
            z = self.activations[layer_index] @ self.weights[layer_index]
            z += self.bias[layer_index]
            self.output_values.append(z)
            a = self.activation_functions[layer_index](z)
            self.activations.append(a)
        return self.activations

    @staticmethod
    def __compute_cost(y_predict: np.ndarray, y_true: np.ndarray):
        """
        This method used to compute loss (cross entropy method) during training.
        :param y_predict is the output from fully connected layer (num_examples x num_classes)
        :param y_true is labels (num_examples x 1)
            Note that y_true is not one-hot encoder vector.
        :return: the loss value with current params of neural network.
        """
        m = y_predict.shape[0]
        network_cost = (1 / m) * np.sum(-y_true * np.log(y_predict) -
                                        (1 - y_true) * np.log(1 - y_predict))
        return network_cost

    @staticmethod
    def __compute_cost_softmax(y_predict, y_true):
        """
        :param y_predict is the output from fully connected layer (num_examples x num_classes)
        :param y_true is labels (num_examples x 1)
            Note that y_true is not one-hot encoded vector.
        """
        m = y_predict.shape[0]
        log_likelihood = -np.log(y_predict[range(m), y_true])
        loss = np.sum(log_likelihood) / m
        return loss

    def __backpropagation(self, y_true: np.ndarray, reg: float):
        """
        This method used to compute and update the params for each epoch
        :param y_true: The target variables
        :return: new param of the neural network
        """
        # Backward
        for index in reversed(range(1, self.number_of_layers)):
            # Backward activation
            if index < self.number_of_layers - 1:
                self.dz[index - 1] = self.backward_functions[index - 1](
                    self.activations[index],
                    self.dz[index].dot(self.weights[index].T))
            else:
                if self.activation_names[-1] == "softmax":
                    self.dz[index - 1] = self.backward_functions[index - 1](self.output_values[index - 1], y_true)
                else:
                    self.dz[-1] = (self.activations[-1] - y_true)  \
                        * self.backward_functions[-1](self.output_values[-1], 1)
            # Backward linear
            self.dw[index - 1], self.db[index - 1] = linear_backward(
                self.activations[index - 1],
                self.weights[index - 1],
                self.dz[index - 1], reg=reg)

    def __update_parameter(self, learning_rate: float):
        """
        :param learning_rate:
        This method used to update to parameter in the each neuron in neural network
        """
        for layer_index in range(self.number_of_layers - 1):
            self.weights[layer_index] -= learning_rate * self.dw[layer_index]
            self.bias[layer_index] -= learning_rate * self.db[layer_index]

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
            self.weights.append(self._init_weights(self.n_units[self.number_of_layers - 2],
                                                   self.n_units[self.number_of_layers - 1]))
            self.bias.append(np.zeros((1, n_unit)))
            self.params.append(self.n_units[self.number_of_layers - 1] * self.n_units[self.number_of_layers - 2])
            self.dz.append(None)
            self.dw.append(None)
            self.db.append(None)
            if activation is None:
                self.activation_names.append("None")
                self.activation_functions.append(None)
            elif activation == "sigmoid":
                self.activation_names.append("sigmoid")
                self.activation_functions.append(sigmoid)
                self.backward_functions.append(sigmoid_backward)
            elif activation == "ReLU":
                self.activation_names.append("ReLU")
                self.activation_functions.append(relu)
                self.backward_functions.append(relu_backward)
            elif activation == "ELU":
                self.activation_names.append("ELU")
                self.activation_functions.append(elu)
                self.backward_functions.append(elu_backward)
            elif activation == "tanh":
                self.activation_names.append("tanh")
                self.activation_functions.append(tanh)
                self.backward_functions.append(tanh_backward)
            elif activation == "LeakyReLU":
                self.activation_names.append("LeakyReLU")
                self.activation_functions.append(leaky_relu)
                self.backward_functions.append(leaky_relu_backward)
            elif activation == "softmax":
                self.activation_names.append("softmax")
                self.activation_functions.append(softmax)
                self.backward_functions.append(softmax_backward)

    @staticmethod
    def batch_generator(x_train: np.ndarray, y_train: np.ndarray, batch_size: int):
        """
        This method used to generate batches to feed to our neural network.
        :param x_train: input variables
        :param y_train: Target variables
        :param batch_size: the size of the batch feed to training our neural network
        :return: Batches of input data
        """
        indices = np.arange(batch_size)
        x_train_batched = []
        y_train_batched = []
        np.random.shuffle(indices)
        for indice in indices:
            x_train_batched.append(x_train[indice])
            y_train_batched.append(y_train[indice])
        return np.array(x_train_batched), np.array(y_train_batched)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
            reg: float = 0, batch_size: int = None, learning_rate: float = 0.001):
        """
        This method used to train the model with x_train and y_train dataset.
        :param x_train: input variables
        :param y_train: target variables
        :param reg: the lambda value used to compute regularization
        :param epochs: the number of iterations to loop the training section.
        :param batch_size: the size of input variables feed to training neural network
        :param learning_rate: the learning rate to update parameter.
        :return: the weights after training
        """
        if batch_size is None:
            batch_size = len(x_train)
        if self.activation_names[-1] == "softmax":
            y_predict = self.__feed_forward(x_train)[-1]
            loss = self.__compute_cost_softmax(y_predict, y_train)
        else:
            y_predict = self.__feed_forward(x_train)[-1].flatten()
            loss = self.__compute_cost(y_predict, y_train)
        reg_loss = reg_loss_calc(self.weights, reg=reg)
        self.history.append(loss + reg_loss)
        print("Epoch: 0 \t Loss: {}".format(loss + reg_loss))
        print("-------------------------------------")
        self.activations = []
        self.output_values = []
        for i in range(epochs):
            batch_x, batch_y = self.batch_generator(x_train, y_train, batch_size)
            self.activations = []
            self.output_values = []
            if self.activation_names[-1] == "softmax":
                self.__feed_forward(batch_x)
                self.__backpropagation(batch_y, reg)
                self.__update_parameter(learning_rate)
                y_predict = self.__feed_forward(batch_x)[-1]
                # y_predict = np.argmax(y_predict, axis=0)
                loss = self.__compute_cost_softmax(y_predict, batch_y)
                self.activations = []
                self.output_values = []
            else:
                for item in range(batch_size):
                    self.__feed_forward(batch_x[item].reshape(1, len(batch_x[item])))
                    self.__backpropagation(batch_y[item], reg)
                    self.__update_parameter(learning_rate)
                    self.activations = []
                    self.output_values = []
                y_predict = self.__feed_forward(batch_x)[-1].flatten()
                loss = self.__compute_cost(y_predict, batch_y)
            reg_loss = reg_loss_calc(self.weights, reg=reg)
            print("Epoch: {} \t Loss: {}".format(i + 1, loss + reg_loss))
            print("-------------------------------------")
            self.history.append(loss + reg_loss)
            self.activations = []
            self.output_values = []
            self.dw = [None] * (self.number_of_layers - 1)
            self.db = [None] * (self.number_of_layers - 1)
            self.dz = [None] * (self.number_of_layers - 1)

    def summary(self):
        """
        This method used to print a string summary of the neural network.
        :return:
        """
        print("Layer \t Activate Function \t params")
        print("=========================================================")
        for layer_index in range(self.number_of_layers - 1):
            print("{} \t {} \t {}".format(self.layer_names[layer_index],
                                          self.activation_names[layer_index],
                                          self.params[layer_index]))
        print("=========================================================")
        print("Total params: ", sum(self.params))

    def pop(self):
        """
        This method used to remove the last layer in the model.
        """
        self.number_of_layers -= 1
        self.activation_names.pop(-1)
        self.params.pop(-1)
        self.n_units.pop(-1)
        self.layer_names.pop(-1)

    def predict(self, x: np.ndarray):
        """
        This method used to predict x through the neural network model.
        :param x: the input variables
        :return: the predict variables
        """
        activations_ = [x]
        z = x.dot(self.weights[0]) + self.bias[0]
        a = self.activation_functions[0](z)
        activations_.append(a)
        for layer_index in range(1, self.number_of_layers - 1):
            z = activations_[layer_index] @ self.weights[layer_index]
            z += self.bias[layer_index]
            a = self.activation_functions[layer_index](z)
            activations_.append(a)
        if self.activation_functions[-1] == softmax:
            activation_last_layer = activations_[-1]
            result = np.argmax(activation_last_layer, axis=1)
        else:
            activation_last_layer = self.__feed_forward(x.reshape(1, len(x)))[-1]
            result = 1 if activation_last_layer[0][0] > 0.5 else 0
        return result

    def accuracy_score(self, y_val: np.ndarray, x_val: np.ndarray):
        """
        This method used to compute the accuracy score of the y_predict
        :param y_val: the actual variables
        :param x_val: the predict value we want to compute score.
        :return: the accuracy score.
        """
        y_predict = self.predict(x_val)
        acc = np.mean(y_predict == y_val)
        print("Train accuracy: {}".format(acc))

    def plot_history(self):
        """
        This method used to plot the loss-history
        """
        plt.plot(self.history, label='train loss')
        plt.legend(loc='best')
        plt.show()


if __name__ == "__main__":
    D = 2  # 2 features
    K = 3  # 3 classes
    N = 3000  # number of samples

    X, y = spiral_gen(int(N / K), D, K)
    X_train, X_test, y_train, y_test = split(X, y, 0.3)
    nn = NeuralNetwork()
    nn.add(layer_name="Dense", activation="sigmoid", n_unit=D)
    nn.add(layer_name="Dense", activation="ReLU", n_unit=256)
    nn.add(layer_name="Dense", activation="ReLU", n_unit=64)
    nn.add(layer_name="Dense", activation="softmax", n_unit=K)
    nn.summary()
    nn.fit(x_train=X, y_train=y, epochs=1000, learning_rate=1, batch_size=2999, reg=0.002)
    plot_decision_boundary(nn.predict, X, y)
    nn.accuracy_score(y_test, X_test)
    nn.plot_history()
