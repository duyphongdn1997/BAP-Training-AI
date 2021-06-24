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

    def compute_cost(self, prediction: np.ndarray, lambda_: float):
        """
        This function takes actual and predicted ratings and compute total mean square 
        error(mse) in observed ratings
        :param prediction: prediction of matrix factorized
        :param _lambda: lambda to regularization
        :return: total mean square error
        """
        mse = 0
        i = len(self.utility_matrix)
        j = len(self.utility_matrix[0])
        for x in range(i):
            for y in range(j):
                if self.utility_matrix[x][y] != 0:
                    error = (lambda_/2) * (self.utility_matrix[x][y] - prediction[x][y]) ** 2
                    mse += error
        return mse

    def get_predicted_ratings(self, users: np.ndarray, items: np.ndarray):
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

    def run_gradient_descent(self, users_: np.ndarray,
                             items_: np.ndarray,
                             iterations: int = 300,
                             lr: float = 0.01,
                             lambda_: float = 0.02,
                             error_limit: float = 0.0001):
        """
        This method used to run gradient descent
        :param error_limit: limited error
        :param lr: learning rate, default = 0.01
        :param iterations: number of iterations to train model
        :param lambda_: lambda to regularization
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
            for x, y, rating_ in sample:
                predict_matrix = self.get_predicted_ratings(users_, items_)
                prediction = predict_matrix[x][y]
                error = rating_ - prediction

                # updating biases
                users_[x] += lr * (2 * error - lambda_ * users_[x])
                items_[y] += lr * (2 * error - lambda_ * items_[y])

                # updating P and Q
                P[x, :] += lr * (2 * error * Q[:, y] - lambda_ * P[x, :])
                Q[:, y] += lr * (2 * error * P[x, :] - lambda_ * Q[:, y])
            mse = self.compute_cost(predict_matrix, lambda_=lambda_)
            err = (i, mse)
            stats.append(err)
            if mse < error_limit:
                break
        """"finally returns (iter,mse) values in a list"""
        # print(stats)
        return stats, P, Q
    
    def matrix_factorization(self, iterations: int = 300, lr: float = 0.001,
                             lambda_: float = 0.02):
        """
        This methods used to get P matrix, Q matrix, predicted utility matrix and plot the loss function.
        :param iterations : the maximum number of steps to perform the optimisation
        :param _lambda: lambda to regularization
        :param lr : the learning rate
        """

        U = np.zeros(self.users)
        I = np.zeros(self.items)
        # Run gradient descent to minimize error
        stats, P, Q = self.run_gradient_descent(users_=U, items_=I, iterations=iterations,
                                          lr=lr, lambda_ = lambda_)

        # print('P matrix:')
        # print(P)
        # print('Q matrix:')
        # print(Q)
        # print("User bias:")
        # print(U)
        # print("Item bias:")
        # print(I)
        print("Utility matrix:")
        print(self.utility_matrix)
        print("Predicted utility matrix:")
        predicted_ratings = self.get_predicted_ratings(U, I)
        for i in predicted_ratings:
            print(i)
        print(len(predicted_ratings))
        self.plot_graph(stats)

    @staticmethod
    def plot_graph(stats):
        """
        This method usedto plot the loss function.
        :param stats: history of the loss function.
        :return:
        """
        i = [i for i, e in stats]
        e = [e for i, e in stats]
        plt.plot(i, e)
        plt.xlabel("Iterations")
        plt.ylabel("Mean Square Error")
        plt.show()


def main():
    anime = pd.read_csv('/content/anime.csv')
    rating = pd.read_csv('/content/rating.csv')
    rating.rating.replace({-1: 0}, inplace = True)
    data_sub = rating[rating['user_id']<=10000]
    data_sub = np.array(data_sub)
    rows, row_pos = np.unique(data_sub[:, 0], return_inverse=True)
    cols, col_pos = np.unique(data_sub[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(rows), len(cols)), dtype=data_sub.dtype)
    pivot_table[row_pos, col_pos] = data_sub[:, 2]
    pivot_table.shape
    model = AnimeRecommendationSystem(utility_matrix=pivot_table)
    model.matrix_factorization(iterations=200, lr=0.001, lambda_=0.02)

if __name__=="__main__":
    main()



