import numpy as np
import matplotlib.pyplot as plt
import activation as func

from deepnet import DeepNet
from utils import shuffle, train_test_split, classification_rate


def donut(count, radius=10, spread=2):
    dots = np.random.rand(count)*np.pi*2

    X = np.zeros((count, 2))

    X[:,0] = np.cos(dots)*radius + (np.random.rand(count) - 0.5)*spread
    X[:,1] = np.sin(dots)*radius + (np.random.rand(count) - 0.5)*spread

    return X


if __name__ == '__main__':
    # Generate some fake data
    N = 2100
    D = 2
    perClass = int(N/3)

    X1 = donut(perClass, radius=10, spread=4)
    X2 = donut(perClass, radius=6, spread=3)
    X3 = donut(perClass, radius=2, spread=4)

    X = np.concatenate((X1, X2, X3), axis=0)
    Y = np.array([0]*perClass + [1]*perClass + [2]*perClass)

    # Normalize the data
    for i in range(D):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()

    # Get train and test sets
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain, Xtest, Ytest = train_test_split(X, Y)

    # Plot the training data
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=Ytrain, cmap="tab10")
    plt.show()

    nn = DeepNet(activation=func.TanH, learning_rate=0.00001, M=5, L=5)
    nn.fit(Xtrain, Ytrain, epochs=50000)
    nn.plot_costs()

    P = nn.predict(Xtest)
    print("Test classification rate:", classification_rate(Ytest, P))

    Xwrong = Xtest[Ytest != P,:]
    Ywrong = Ytest[Ytest != P]

    # Plot the predicted data
    plt.scatter(Xtest[:,0], Xtest[:,1], c=P, cmap="tab10")
    plt.scatter(Xwrong[:,0], Xwrong[:,1], s=150, c=Ywrong, cmap="tab10", marker="x", linewidths=2)

    plt.show()
