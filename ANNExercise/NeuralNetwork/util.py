"""
These code copied from Andrej Karpathy, on his cs231n course github code.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def spiral_gen(N, D, K):
    # N number of points per class
    # D dimensionality
    # K number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='int8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    return X, y


def reg_loss_calc(w_list, l2=True, reg=1e-3):
    loss = 0
    if l2:
        for i in range(len(w_list)):
            loss += 0.5 * reg * np.sum(w_list[i] * w_list[i])
    else:
        for i in range(len(w_list)):
            loss += 0.5 * reg * np.sum(abs(w_list[i]))
    return loss


def split(X, y, test_proportion):
    ratio = int(X.shape[0] * test_proportion)
    X_train = X[ratio:, :]
    X_test = X[:ratio, :]
    Y_train = y[ratio:]
    Y_test = y[:ratio]
    return X_train, X_test, Y_train, Y_test
