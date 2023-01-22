import numpy as np

def sigmoid(x, a=1):
    x = np.array([float(b) for b in x]) #float conversion of the vector x to avoid having a 0 integer in the np.exp argument
    return 1 / (1 + np.exp(-(a * x)))


def sigmoid_derivative(x, a=1):
    return a * sigmoid(x, a=a) * (1 - sigmoid(x, a=a))


def relu(x, a=None):
    return np.maximum(0, x)


def relu_derivative(x, a=None):
    return (np.sign(x)+1)/2


def act_f(label, k):
    """return the activation functions with k=0,
    their derivative with k=1 """
    if label == "sigmoid" and k == 0:
        return sigmoid

    if label == "sigmoid" and k == 1:
        return sigmoid_derivative

    if label == "relu" and k == 0:
        return relu

    if label == "relu" and k == 1:
        return relu_derivative