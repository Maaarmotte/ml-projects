import numpy as np


class ActivationFunction:
    @staticmethod
    def apply(Z):
        pass

    @staticmethod
    def apply_derivative(Z):
        pass


class Sigmoid(ActivationFunction):
    @staticmethod
    def apply(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def apply_derivative(Z):
        return Z*(1 - Z)


class TanH(ActivationFunction):
    @staticmethod
    def apply(Z):
        return np.tanh(Z)

    @staticmethod
    def apply_derivative(Z):
        return 1 - Z*Z

class ReLU(ActivationFunction):
    @staticmethod
    def apply(Z):
        return np.maximum(0, Z)

    @staticmethod
    def apply_derivative(Z):
        return np.sign(Z)
