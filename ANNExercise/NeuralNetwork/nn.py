import numpy as np
import random
from util import *


def reg_loss_calc(W_list, l2=True, reg=1e-3):
    loss = 0
    if l2:
        for i in range(len(W_list)):
            loss += 0.5 * reg * np.sum(W_list[i] * W_list[i])
    else:
        for i in range(len(W_list)):
            loss += 0.5 * reg * np.sum(abs(W_list[i]))
    return loss


class Softmax:
    def forward(self, scores):
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def backward(self, scores, y):
        N = scores.shape[0]
        dscores = self.forward(scores)
        dscores[range(N), y] -= 1
        dscores /= N
        return (dscores)

    def cross_entropy(self, probs, y):
        # probs: N x K
        N = probs.shape[0]
        # print('All normalized probs:',probs.shape)

        # normalized prob of correct label
        correct_logprobs = -np.log(probs[range(N), y])
        # print('-log(normalized probs) when label is correct:',correct_logprobs.shape)

        # data loss
        data_loss = np.sum(correct_logprobs) / N

        return data_loss


class Norm:
    def forward(self, scores, epsilon=1e-3):
        self.mu = np.mean(scores, axis=0)
        self.sd = np.sqrt(np.var(scores + epsilon, axis=0))
        self.N = scores.shape[0]
        normalized = (scores - self.mu) / self.sd
        return (normalized)

    def backward(self, scores, previous, epsilon=1e-3):
        out = self.forward(scores, epsilon)
        dnormalized = (1 - (1.0 / self.N)) * (
                self.sd - (2 * np.sqrt(np.var(scores, axis=0)) * (scores - self.mu) ** 2) / (self.N * self.sd))
        dnormalized *= previous
        return dnormalized


class Dropout:
    def __init__(self, p, seed):
        self.seed = seed
        self.p = p

    def mask(self, scores, scaling=True):
        # if no dropout
        if self.p == 1:
            return np.ones_like(scores)
        np.random.seed(seed=self.seed)
        if scaling:
            scaler = 1.0 / (1 - self.p)
        else:
            scaler = 1.0
        mask = scaler * np.random.binomial(n=1, p=self.p, size=scores.shape)
        return mask


class ReLu:
    def forward(self, scores):
        # scores: N x K
        out = scores * (scores > 0)
        return out

    def backward(self, scores, previous):
        dscores = 1.0 * (scores > 0) * previous
        return (dscores)


class Tanh:
    def forward(self, scores):
        # scores: N x K
        out = np.tanh(scores)
        return out

    def backward(self, scores, previous):
        out = self.forward(scores)
        dscores = (1.0 - np.square(out)) * previous
        return (dscores)


class Sigmoid:

    def forward(self, scores):
        # scores: N x K
        out = 1.0 / (1.0 + np.exp(-scores))
        return out

    def backward(self, scores, previous):
        out = self.forward(scores)
        dscores = (1.0 - out) * out * previous
        return (dscores)


class Linear:
    def forward(self, X, W, b):
        # X: N x D
        # W: D x K
        scores = X.dot(W) + b  # N x K
        return (scores)

    def backward(self, X, W, previous, reg=1e-3):
        # previous: N x K
        dW = X.T.dot(previous)
        dW += reg * W
        db = np.sum(previous, axis=0, keepdims=True)
        return dW, db


def init_weights(D, K, h_list, scaling, const=0.1):
    nb_layers = len(h_list) + 1
    W_list = [None] * nb_layers
    b_list = [None] * nb_layers

    # randomize weights
    for i in range(nb_layers):
        # scaling term
        if scaling == 'constant':
            scaler = const
        elif scaling == 'xavier':
            scaler = np.sqrt(1.0 / h_list[i - 1])
        elif scaling == 'he':
            scaler = np.sqrt(2.0 / h_list[i - 1])
        else:
            raise ValueError('Can only use "constant","xavier" or "he" scaling')

        # input layer
        if (i == 0):
            W_list[i] = np.random.randn(D, h_list[i])
            b_list[i] = np.zeros((1, h_list[i]))
        # output layer
        elif (i == nb_layers - 1):
            W_list[i] = np.random.randn(h_list[i - 1], K) * scaler
            b_list[i] = np.zeros((1, K))
        # hidden layers
        else:
            W_list[i] = np.random.randn(h_list[i - 1], h_list[i]) * scaler
            b_list[i] = np.zeros((1, h_list[i]))
    return W_list, b_list


class DeepModel:
    def __init__(self, X, y, D, K, h_list, p_list,
                 activation_list, output_activation, scaling, const=0.1):
        # parameters
        self.X = X
        self.y = y
        self.nb_layers = len(h_list) + 1
        nb_layers = self.nb_layers
        self.h_list = h_list
        self.p_list = p_list
        self.W_list = [None] * nb_layers
        self.b_list = [None] * nb_layers
        self.X_list = [None] * nb_layers
        self.X_list[0] = self.X

        self.act_list = [None] * nb_layers
        self.lin_list = [None] * nb_layers
        self.drop_list = [None] * nb_layers
        self.linout_list = [None] * nb_layers
        self.actout_list = [None] * nb_layers
        self.mask_list = [None] * nb_layers

        self.dscores_list = [None] * nb_layers
        self.dW_list = [None] * nb_layers
        self.db_list = [None] * nb_layers

        # randomize weights
        for i in range(nb_layers):
            # scaling term
            if scaling == 'constant':
                scaler = const
            elif scaling == 'xavier':
                scaler = np.sqrt(1.0 / h_list[i - 1])
            elif scaling == 'he':
                scaler = np.sqrt(2.0 / h_list[i - 1])
            else:
                raise ValueError('Can only use "constant","xavier" or "he" scaling')
            # linear layers
            self.lin_list[i] = Linear()
            if (i == 0):
                self.W_list[i] = np.random.randn(D, h_list[i])
                self.b_list[i] = np.zeros((1, h_list[i]))
            elif (i == nb_layers - 1):
                self.W_list[i] = np.random.randn(h_list[i - 1], K) * scaler
                self.b_list[i] = np.zeros((1, K))
            else:
                self.W_list[i] = np.random.randn(h_list[i - 1], h_list[i]) * scaler
                self.b_list[i] = np.zeros((1, h_list[i]))

            # activation layers
            seed = 1
            if (i < nb_layers - 1):
                self.act_list[i] = activation_list[i]
                self.drop_list[i] = Dropout(seed=seed, p=p_list[i])
            else:
                self.act_list[i] = output_activation
                self.drop_list[i] = Dropout(seed=seed, p=1)

    def summary(self):
        nb_layers = self.nb_layers
        h_list = self.h_list
        N = self.X_list[0].shape[0]

        for i in range(nb_layers):

            print('[LIN] {}: Input ({},{}), Weight {}, Bias {}, Output ({},{})'.format(i,
                                                                                       N, self.W_list[i].shape[0],
                                                                                       self.W_list[i].shape,
                                                                                       self.b_list[i].shape,
                                                                                       N, self.W_list[i].shape[1]))

            if (i < nb_layers - 1):
                print('[ACT] {}: Input-Output ({},{}), Dropout {}, Activation {}'.format(i, N, self.W_list[i].shape[1], \
                                                                                         self.p_list[i],
                                                                                         self.act_list[i]))
            else:
                print('[OUT] {}: Input-Output ({},{})'.format(i, N, self.W_list[i].shape[1]))

    def train(self, iter, reg, step_size, use_l2):
        nb_layers = self.nb_layers
        X_list = self.X_list
        act_list = self.act_list
        lin_list = self.lin_list
        actout_list = self.actout_list
        linout_list = self.linout_list
        dscores_list = self.dscores_list
        dW_list = self.dW_list
        db_list = self.db_list
        y = self.y

        for j in np.arange(iter):
            # forward
            for i in range(nb_layers):
                linout_list[i] = lin_list[i].forward(X_list[i], self.W_list[i], self.b_list[i])
                actout_list[i] = act_list[i].forward(linout_list[i])
                # dropout
                self.mask_list[i] = self.drop_list[i].mask(actout_list[i])
                actout_list[i] *= self.mask_list[i]
                if (i < nb_layers - 1): X_list[i + 1] = actout_list[i]

            if type(act_list[nb_layers - 1]) is Softmax:
                data_loss = act_list[nb_layers - 1].cross_entropy(actout_list[nb_layers - 1], y)
            else:
                data_loss = act_list[nb_layers - 1].hinge(actout_list[nb_layers - 1], y)
            reg_loss = reg_loss_calc(self.W_list, l2=use_l2, reg=reg)
            loss = data_loss + reg_loss

            if j % (iter / 10) == 0:
                print('Loss at iteration {}: {}'.format(j, loss))

            # backward
            for i in reversed(range(nb_layers)):
                # backward activation
                if i < nb_layers - 1:
                    dscores_list[i] = act_list[i].backward(X_list[i + 1], dscores_list[i + 1].dot(self.W_list[i + 1].T))
                    # dropout
                    dscores_list[i] *= self.mask_list[i]
                else:
                    dscores_list[i] = act_list[i].backward(linout_list[i], y)

                    # backward linear
                dW_list[i], db_list[i] = lin_list[i].backward(X_list[i], self.W_list[i], dscores_list[i], reg=reg)

            # update
            for i in range(nb_layers):
                self.W_list[i] -= step_size * dW_list[i]
                self.b_list[i] -= step_size * db_list[i]

    def pred(self, X):
        nb_layers = self.nb_layers
        X_list = self.X_list
        X_list[0] = X
        act_list = self.act_list
        lin_list = self.lin_list
        actout_list = self.actout_list
        linout_list = self.linout_list
        W_list = self.W_list
        b_list = self.b_list

        for i in range(nb_layers):
            linout_list[i] = lin_list[i].forward(X_list[i], W_list[i], b_list[i])
            actout_list[i] = act_list[i].forward(linout_list[i])
            # if i < nb_layers - 1:
            #     X_list[i + 1] = actout_list[i]
        pred = np.argmax(actout_list[nb_layers - 1], axis=1)
        return pred


D = 2  # 2 features
K = 3  # 3 classes
N = 3000  # number of samples

X, y = spiral_gen(int(N / K), D, K)
X_train, X_test, y_train, y_test = split(X, y, 0.3)
mod = DeepModel(X, y, D, K, h_list=[100, 50], p_list=[1, 1],
                activation_list=[ReLu(), ReLu()],
                output_activation=Softmax(), scaling='he')

mod.train(iter=1000, reg=0.02, step_size=1, use_l2=True)
# print('Accuracy: {}'.format(np.mean(y == mod.pred(X))))
plot_decision_boundary(mod.pred, X, y)
