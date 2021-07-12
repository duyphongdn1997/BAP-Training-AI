import numpy as np


def sigmoid(z):
    """
    This method used to compute sigmoid function
    :param: z: input of sigmoid function
    :return: result after use sigmoid function to z.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_backward(z, in_put):
    """
    This method used to compute back ward of the Sigmoid activation functions
    :param z: activation value of the next layer
    :param in_put: The result of dz(l+1) @ weight(l+1)
    :return:
    """
    sigmoid_ = 1.0 / (1.0 + np.exp(-z))
    dz = sigmoid_ * (1.0 - sigmoid_) * in_put
    return dz


def relu(z):
    """
    This method used to compute the ReLU function
    :param z: input variables
    :return: Output of ReLU function
    """
    return np.maximum(0, z)


def relu_backward(z, in_put):
    """
    This method used to compute backward of the ReLU activation functions
    :param z: activation value of the next layer
    :param in_put: The result of dz(l+1) @ weight(l+1)
    :return: dz to compute backward of the previous layer.
    """
    dz = np.clip(z > 0, 0, 1.0)
    dz *= in_put
    return dz


def leaky_relu(z, alpha: float = 0.01):
    """
    This method used to compute the leaky ReLU function
    :param z: input variables
    :param alpha: input variables
    :return: Output of Leaky ReLU function
    """
    return np.maximum(alpha * z, z)


def leaky_relu_backward(z, in_put, alpha: float = 0.01):
    """
    This method used to compute the backward of the LeakyReLU activation functions.
    :param alpha: input variables
    :param z: activation value of the next layer
    :param in_put: The result of dz(l+1) @ weight(l+1)
    """
    dz = np.clip(z > 0, alpha, 1.0)
    dz *= in_put
    return dz


def elu(z, alpha: float = 1):
    """
    This method used to compute the ELU function
    :param z: input variables
    :param alpha: input variables
    :return: Output of ELU function
    """
    return np.maximum(alpha * (np.exp(z) - 1), z)


def elu_backward(z, in_put, alpha: float = 1):
    """
    This method used to compute backward of ELU activation function.
    :param alpha: input variable
    :param z: activation value of the next layer
    :param in_put: The result of dz(l+1) @ weight(l+1)
    :return:
    """
    dz = np.clip(z > 0, alpha * np.exp(z), 1.0)
    dz *= in_put
    return dz


def tanh(z):
    """
    This method used to compute the tanh function
    :param z: input variables
    :return: Output of tanh function
    """
    return np.tanh(z)


def tanh_backward(z, in_put):
    """
    This method used to compute the backward of the Tanh activation function
    :param z: activation value of the next layer
    :param in_put: The result of dz(l+1) @ weight(l+1)
    :return:
    """
    out = tanh(z)
    dz = (1.0 - np.square(out)) * in_put
    return dz


def linear_backward(A, w, dz, reg):
    """
    This method used to compute the backward of the Layer.
    :param A: The activations value
    :param w: The weights matrix of the layer
    :param dz: The backward value of the activation function
    :param reg: The value to compute regularization
    :return: dW, db to update our parameters
    """
    dW = A.T.dot(dz)
    # dW = dz.dot(A.T)
    dW += reg * w
    db = np.sum(dz, axis=0, keepdims=True)
    return dW, db


def softmax(z):
    """
    This method used to compute the softmax function
    Note: For only last layer
    :param z: the input variables
    :return: The probabilities of X belong to each class.
    """
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return probs


def softmax_backward(z, y):
    """
    This method used to compute backward of the Softmax activation functions
    :param z: The activations value after softmax.
    :param y: The actual value.
    :return: dz to compute backward of the previous layer.
    """
    N = z.shape[0]
    dz = softmax(z)
    dz[range(N), y] -= 1
    dz /= N
    return dz
