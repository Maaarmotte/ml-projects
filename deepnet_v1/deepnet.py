import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
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
    mu = 0  # Nesterov momentum

    activation = None   # Activation function
    weights = []
    bias = []
    train_costs = []

    def __init__(self, activation, L=3, M=3, learning_rate=0.001, l1=0, l2=0, momentum=0):
        self.activation = activation
        self.L = L
        self.M = M
        self.r = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.mu = momentum

    def fit(self, X, Y, epochs=100000, batch_size=0, print_interval=5):
        self.N, self.D = X.shape
        self.K = np.max(Y) + 1

        # Initialize random weights. We suppose that the data is normalized so we divide by sqrt(D)
        Ws = [np.random.randn(self.D, self.M) / np.sqrt(self.D)]
        bs = [np.random.randn(self.M)]

        Ws += [np.random.randn(self.M, self.M) / np.sqrt(self.D) for _ in range(self.L-2)]
        bs += [np.random.randn(self.M) for _ in range(self.L-2)]

        Ws += [np.random.randn(self.M, self.K) / np.sqrt(self.D)]
        bs += [np.random.randn(self.K)]

        # Velocities for momentum
        Wvs = [0]*self.L
        bvs = [0]*self.L

        # Indicator matrix: (N, 1) -> (N, K)
        Yk = one_hot_encode(Y)

        # Prepare the batches
        if batch_size <= 0:
            B = 1
            batch_size = len(Y)
        else:
            B = int(np.ceil(len(Y) / batch_size))

        lt = datetime.now()
        for epoch in range(epochs):
            # For all batches
            for b in range(B):
                Xb = X[b*batch_size:(b + 1)*batch_size]
                Yb = Yk[b*batch_size:(b + 1)*batch_size]

                P, XZs = self.__forward(Xb, Ws, bs)

                nWs = [0]*self.L
                nbs = [0]*self.L
                nWvs = [0]*self.L
                nbvs = [0]*self.L

                # For all layers
                for i in range(self.L - 1, -1, -1):
                    w = Ws[i]
                    b = bs[i]
                    wv = Wvs[i]
                    bv = bvs[i]

                    dw, db = self.__gradient(Yb, P, XZs, Ws, i)

                    # Apply the learning rate & regularization
                    # We divide by the batch size so the learning rate doesn't depend on the number of samples
                    upW = self.r*((dw / batch_size) + self.l2*w + self.l1*np.sign(w))
                    upb = self.r*((db / batch_size) + self.l2*b + self.l1*np.sign(b))

                    # Apply momentum and do the weight update
                    # We use momentum to avoid getting stuck in directions that are almost flat
                    nWvs[i] = self.mu*wv - upW
                    nbvs[i] = self.mu*bv - upb

                    nWs[i] = w + self.mu*nWvs[i] - upW
                    nbs[i] = b + self.mu*nbvs[i] - upb

                Ws = nWs
                bs = nbs
                Wvs = nWvs
                bvs = nbvs

                # Print the current progress every X seconds
                if datetime.now() - timedelta(seconds=print_interval) > lt:
                    lt = datetime.now()
                    cost = categorical_cross_entropy(Yb, P)
                    self.train_costs.append(cost)
                    print("Cost:", cost)
                    print("Classification rate", classification_rate(np.nanargmax(Yb, axis=1), np.nanargmax(P, axis=1)))

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
