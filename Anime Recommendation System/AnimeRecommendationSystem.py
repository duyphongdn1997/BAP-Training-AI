import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AnimeRecommendationSystem:
    """
    Anime-recommendation-system
    """

    def __init__(self, utility_matrix, k: int = 3):
        """

        :param utility_matrix
        :param k: latent features count
        """
        self.utility_matrix = utility_matrix
        self.k = k
        self.users, self.items = utility_matrix.shape
        # Create P an initial matrix of dimension N x K
        # Create Q an initial matrix of dimension M x K
        self.P = np.random.rand(self.users, k)
        self.Q = np.random.rand(self.items, k)
        # Get matrix of dimension K x M
        self.Q = self.Q.T

    def __compute_cost(self, prediction):
        """
        This function takes actual and predicted ratings and compute total mean square error(mse) in observed ratings
        :param prediction: prediction of matrix factorized
        :return: total mean square error
        """
        mse = 0
        i = len(self.utility_matrix)
        j = len(self.utility_matrix[0])
        for x in range(i):
            for y in range(j):
                if self.utility_matrix[x][y] != 0:
                    error = (self.utility_matrix[x][y] - prediction[x][y]) ** 2
                    mse += error
        return mse

    def __get_predicted_ratings(self, users, items):
        """
        This method takes P(m*k) and Q(k*n) matrices along with user bias (U)
        and item bias (I) and returns predicted rating.
        :param users: User bias
        :param items: Item bias
        :return:
        """

        m = self.items
        n = self.users
        matrix = []
        for i in range(n):
            row = []
            for j in range(m):
                dot = self.P[i, :].dot(self.Q[:, j])
                predicted = users[i] + items[j] + dot
                row.append(predicted)
            matrix.append(row)
        return matrix

    def __run_gradient_descent(self, users_, items_,
                               iterations: int = 300,
                               lr: float = 0.01,
                               error_limit: float = 0.0001):
        """
        :param error_limit: limited error
        :param lr: learning rate, default = 0.01
        :param iterations: number of iterations to train model
        :param users_:
        :param items_:
        :return:
        """
        global predict_matrix
        stats = []
        sample = []
        n = len(self.utility_matrix)
        m = len(self.utility_matrix[0])
        for x in range(n):
            for y in range(m):
                # contains row, col and value of utility matrix's entry != 0
                if self.utility_matrix[x][y] != 0:
                    tup = (x, y, self.utility_matrix[x][y])
                    sample.append(tup)

        P = self.P
        Q = self.Q

        for i in range(iterations):
            for x, y, r in sample:
                predict_matrix = self.__get_predicted_ratings(users_, items_)
                prediction = predict_matrix[x][y]
                error = r - prediction

                # updating biases
                users_[x] += lr * (2 * error)
                items_[y] += lr * (2 * error)

                # updating P and Q
                P[x, :] += lr * (2 * error * Q[:, y])
                Q[:, y] += lr * (2 * error * P[x, :])
            mse = self.__compute_cost(predict_matrix)
            err = (i, mse)
            stats.append(err)

            if mse < error_limit:
                break
        """"finally returns (iter,mse) values in a list"""
        # print(stats)
        return stats

    def matrix_factorization(self, iterations: int = 300, lr: float = 0.001):
        """
        :param iterations : the maximum number of steps to perform the optimisation
        :param lr : the learning rate
        """

        U = np.zeros(self.users)
        I = np.zeros(self.items)
        # Run gradient descent to minimize error
        stats = self.__run_gradient_descent(users_=U, items_=I, iterations=iterations, lr=lr)

        print('P matrix:')
        print(self.P)
        print('Q matrix:')
        print(self.Q)
        print("User bias:")
        print(U)
        print("Item bias:")
        print(I)
        print("P x Q:")
        print(self.__get_predicted_ratings(U, I))
        print(len(self.__get_predicted_ratings(U, I)))
        self.plot_graph(stats)

    @staticmethod
    def plot_graph(stats):
        """
        This methods used to plot the loss stored when training/updating P/Q process
        :param stats: History of loss
        :return:
        """
        i = [i for i, e in stats]
        e = [e for i, e in stats]
        plt.plot(i, e)
        plt.xlabel("Iterations")
        plt.ylabel("Mean Square Error")
        plt.show()
