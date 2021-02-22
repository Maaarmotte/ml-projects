import numpy as np


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def categorical_cross_entropy(Y, P):
    return -(Y*np.log(P)).sum()


def one_hot_encode(Y):
    N = len(Y)
    max = np.max(Y)

    Yk = np.zeros((N,max+1))
    for i in range(N):
        Yk[i,Y[i]] = 1

    return Yk


def classification_rate(Y, P):
    return np.array(Y == P).astype(int).mean()


def shuffle(X, Y):
    idx = np.arange(len(Y))
    np.random.shuffle(idx)
    return X[idx], Y[idx]


# Outputs trainX, trainY, testX, testY
def train_test_split(X, Y, ratio=0.85):
    idx = int(ratio*len(Y))
    return X[:idx], Y[:idx], X[idx:], Y[idx:]
