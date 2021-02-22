import numpy as np
import matplotlib.pyplot as plt

from utils import softmax, categorical_cross_entropy, one_hot_encode, classification_rate


class DeepNet:
    N = 0   # Number of samples
    D = 0   # Number of features
    L = 0   # Number of layers
    M = 0   # Number of hidden units per layer
    K = 0   # Number of output classes
    r = 0   # Learning rate
    l1 = 0  # L1 regularization factor (LASSO)
    l2 = 0  # L2 regularization factor (Ridge)

    activation = None   # Activation function
    weights = []
    bias = []
    train_costs = []

    def __init__(self, activation, L=3, M=3, learning_rate=0.001, l1=0, l2=0):
        self.activation = activation
        self.L = L
        self.M = M
        self.r = learning_rate
        self.l1 = l1
        self.l2 = l2

    def fit(self, X, Y, epochs=100000):
        self.N, self.D = X.shape
        self.K = np.max(Y) + 1

        # Initialize random weights. We suppose that the data is normalized so we divide by sqrt(D)
        Ws = [np.random.randn(self.D, self.M) / np.sqrt(self.D)]
        bs = [np.random.randn(self.M)]

        Ws += [np.random.randn(self.M, self.M) / np.sqrt(self.D) for _ in range(self.L-2)]
        bs += [np.random.randn(self.M) for _ in range(self.L-2)]

        Ws += [np.random.randn(self.M, self.K) / np.sqrt(self.D)]
        bs += [np.random.randn(self.K)]

        # Indicator matrix: (N, 1) -> (N, K)
        Yk = one_hot_encode(Y)

        for epoch in range(epochs):
            P, XZs = self.__forward(X, Ws, bs)

            if epoch % 1000 == 0:
                cost = categorical_cross_entropy(Yk, P)
                self.train_costs.append(cost)
                print("Cost:", cost)
                print("Classification rate", classification_rate(Y, np.nanargmax(P, axis=1)))

            nWs = [0]*self.L
            nbs = [0]*self.L

            for i in range(self.L - 1, -1, -1):
                w = Ws[i]
                b = bs[i]

                dw, db = self.__gradient(Yk, P, XZs, Ws, i)

                nWs[i] = w - self.r*(dw + self.l2*w + self.l1*np.sign(w))
                nbs[i] = b - self.r*(db + self.l2*b + self.l1*np.sign(b))

            Ws = nWs
            bs = nbs

        self.weights = Ws
        self.bias = bs

    def predict(self, X):
        P, XZs = self.__forward(X, self.weights, self.bias)
        return np.nanargmax(P, axis=1)

    def plot_costs(self):
        plt.plot(self.train_costs)
        plt.show()

    def __forward(self, X, Ws, bs):
        XZs = [X]
        for i in range(len(Ws) - 1):
            XZs.append(self.activation.apply(XZs[i].dot(Ws[i]) + bs[i]))

        return softmax(XZs[-1].dot(Ws[-1]) + bs[-1]), XZs

    def __gradient(self, Y, P, XZs, Ws, layer):
        output = P - Y

        for i in range(self.L - 1, layer, -1):
            output = output.dot(Ws[i].T) * self.activation.apply_derivative(XZs[i])

        return XZs[layer].T.dot(output), output.sum(axis=0)
