import numpy as np


def sigmoid(z):
    """
    This method used to compute sigmoid function
    :param: z: input of sigmoid function
    :return: result after use sigmoid function to z.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(z, in_put):
    a = 1.0 / (1.0 + np.exp(-z))
    dz = a * (1.0 - a) * in_put
    return dz


def relu(z):
    """
    This method used to compute the ReLU function
    :param z: input variables
    :return: Output of ReLU function
    """
    return z if z > 0 else 0


def relu_backward(z, in_put):
    pass


def leaky_relu(z, alpha: float = 0.01):
    """
    This method used to compute the leaky ReLU function
    :param z: input variables
    :param alpha: input variables
    :return: Output of Leaky ReLU function
    """
    return z if z > 0 else alpha * z


def leaky_relu_backward(z, in_put):
    pass


def elu(z, alpha: float = 1):
    """
    This method used to compute the ELU function
    :param z: input variables
    :param alpha: input variables
    :return: Output of ELU function
    """
    return z if z > 0 else alpha * (np.exp(z) - 1)


def elu_backward(z, in_put):
    pass


def tanh(z):
    """
    This method used to compute the tanh function
    :param z: input variables
    :return: Output of tanh function
    """
    return np.tanh(z)


def tanh_backward(z, in_put):
    pass


def linear_backward(z, in_put):
    pass