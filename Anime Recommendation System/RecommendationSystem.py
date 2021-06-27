import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RecommendationSystem(object):
    """
    This is class RecommendationSystem implement matrix factorized based on Exercise 8 in
    Machine Learning course.
    """

    def __init__(self, R: np.ndarray, k: int = 3):
        """
        This is the constructor
        :param R: The matrix where R(i,j) = 1 if user "j" five rating to the anime "i"
        :param k: The number of latent factor
        """
        self.R = R
        self.k = k

    def cofi_cost_function(self, params: np.ndarray, Y: np.ndarray, R: np.ndarray,
                           num_users: int, num_anime: int,
                           k: int = 3, _lambda: float = 0.02):
        """
        This method used to compute the cost and gradient for matrix factorized.
        :param params:
        :param Y:
        :param R:
        :param num_users:
        :param num_anime:
        :param k:
        :param _lambda:
        :return:
        """
        # Unpack values
        X = params[:num_anime * k].reshape((num_anime, k))
        Theta = params[num_anime * k:].reshape((num_users, k))
        # Compute error actual vs predict
        predictions = X @ Theta.T
        err = (predictions - Y)
        J = 1 / 2 * np.sum((err ** 2) * R)

        # compute regularized cost function
        reg_X = _lambda / 2 * np.sum(Theta ** 2)
        reg_Theta = _lambda / 2 * np.sum(X ** 2)
        reg_J = J + reg_X + reg_Theta

        X_grad = err * R @ Theta
        Theta_grad = (err * R).T @ X
        grad = np.append(X_grad.flatten(), Theta_grad.flatten())

        # Compute regularized gradient
        reg_X_grad = X_grad + _lambda * X
        reg_Theta_grad = Theta_grad + _lambda * Theta
        reg_grad = np.append(reg_X_grad.flatten(), reg_Theta_grad.flatten())

        return reg_J, reg_grad

    def normalize_rating(self, Y: np.ndarray, R: np.ndarray):
        """
        This method used to normalize rating R matrix and Y matrix
        :param Y:
        :param R:
        :return:
        """
        m, n = Y.shape[0], Y.shape[1]
        Ymean = np.zeros((m, 1))
        Ynorm = np.zeros((m, n))

        for i in range(m):
            Ymean[i] = np.sum(Y[i, :]) / np.count_nonzero(R[i, :])
            Ynorm[i, R[i, :] == 1] = Y[i, R[i, :] == 1] - Ymean[i]

        return Ynorm, Ymean

    def gradient_descent(self, initial_parameters, Y, R,
                         num_users, num_anime, num_features,
                         alpha, num_iters, Lambda):
        """
        Optimize X and Theta
        """
        # unfold the parameters
        X = initial_parameters[:num_anime * num_features].reshape(num_anime, num_features)
        Theta = initial_parameters[num_anime * num_features:].reshape(num_users, num_features)

        J_history = []

        for i in range(num_iters):
            params = np.append(X.flatten(), Theta.flatten())
            cost, grad = self.cofi_cost_function(params, Y, R, num_users, num_anime, num_features, Lambda)

            # unfold grad
            X_grad = grad[:num_anime * num_features].reshape(num_anime, num_features)
            Theta_grad = grad[num_anime * num_features:].reshape(num_users, num_features)
            X = X - (alpha * X_grad)
            Theta = Theta - (alpha * Theta_grad)
            J_history.append(cost)

        paramsFinal = np.append(X.flatten(), Theta.flatten())
        return paramsFinal, J_history

    def plot_cost_function(self, J_history):
        """
        This method used to plot the cost function
        :param J_history: stored the cost function for each iter.
        :return:
        """
        plt.plot(J_history)
        plt.xlabel("Iteration")
        plt.ylabel("$J (Theta)$")
        plt.title("Cost function using Gradient Descent")
