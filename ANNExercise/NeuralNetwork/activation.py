import numpy as np


class ReLu:

    def forward(self, scores):
        #scores: N x K
        out = scores * (scores > 0)
        return out

    def backward(self, scores, previous):
        d_scores = 1.0 * (scores > 0) * previous
        return d_scores


class Sigmoid:

    def forward(self, scores):
        # scores: N x K
        out = 1.0 / (1.0 + np.exp(-scores))
        return out

    def backward(self, scores, previous):
        out = self.forward(scores)
        d_scores = (1.0 - out) * out * previous
        return d_scores

def sigmoid_backward(z, input):

    a = 1.0 / (1.0 + np.exp(-z))
    dz = a * (1.0 - a) * input
    return dz

class Tanh:

    def forward(self, scores):
        # scores: N x K
        out = np.tanh(scores)
        return out

    def backward(self, scores, previous):
        out = self.forward(scores)
        d_scores = (1.0 - np.square(out)) * previous
        return d_scores


