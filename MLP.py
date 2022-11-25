import numpy as np
import pandas as pd
import matplotlib as plt
import math as math


def sigmoid(x, a=1):
    return 1 / (1 + np.exp(-(a * x)))


def sigmoid_derivative(x, a=1):
    return a * sigmoid(x, a) * (1 - sigmoid(x, a))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (np.sign(x)+1)/2


def act_f(label, k):
    """return the activation functions with k=0 and their derivative with k=1 """
    if label == "sigmoid" and k == 0:
        return sigmoid

    if label == "sigmoid" and k == 1:
        return sigmoid_derivative

    if label == "relu" and k == 0:
        return relu

    if label == "relu" and k == 1:
        return relu_derivative

    
class MLP:

    def __init__(self, sizes, activation):
        """ sizes is a list of dimensions (number of units for each layer)
            activation is a string (sigmoid/relu) related to activation function
            used in the hidden units"""
        self.sizes = sizes
        self.activation = activation
        self.n_input = sizes[0]  # number of input units
        self.n_output = sizes[-1]  # number of output units
        self.hidden_layers = np.array([np.random.randn(y, x) for (y, x) in zip(self.sizes[1:], self.sizes[:-1])], dtype=object)  # list of weight matrices for hidden layers
        self.biases = np.array([np.random.random(x) for x in self.sizes[1:]], dtype=object)

    def feedforward(self, input):
        for (W, b) in zip(self.hidden_layers[:-1], self.biases[:-1]):
            input = act_f(self.activation, 0)(W @ input + b)
        net = self.hidden_layers[-1].dot(input) + self.biases[-1]  # compute the net for the output layer
        output = sigmoid(net)  # compute the output for the output layer
        return output

    def saved_feedforward(self, input):
        output = [input]
        net_matrix = []
        for i, (W, b) in enumerate(zip(self.hidden_layers[:-1], self.biases[:-1])):
            net = W.dot(input) + b  # compute the net for a hidden layer
            o = act_f(self.activation, 0)(net)  # compute the output for a hidden layer
            input = o  # assigning inputs
            output.append(o)  # save outputs
            net_matrix.append(net)  # save nets
        net = self.hidden_layers[-1].dot(input) + self.biases[-1]  # compute the net for the output layer
        output.append(sigmoid(net))  # compute the output for the output layer
        net_matrix.append(net)
        return output, net_matrix

    def backprop(self, input, target, o, net, grad_list, grad_bias_list, i=0):
        """Compute the derivatives of the MSE cost function wrt weights and biases """
        if i == (len(self.hidden_layers) - 1):  # last layer
            delta_k = (target - o[1]) * sigmoid_derivative(net[0])  # compute the final delta (delta_L)
            derivatives = np.outer(delta_k, o[0])  # compute the derivatives wrt last weights, here o[0]=o_(L-1)
            grad_list.append(derivatives)
            grad_bias_list.append(delta_k)  # append derivatives wrt biases that coincides with delta_L
            return delta_k, grad_list, grad_bias_list
        i += 1
        delta_j = self.backprop(input, target, o[1:], net[1:], grad_list, grad_bias_list, i)[0]  # recursive call (delta_(i+1))
        pre_delta = self.hidden_layers[i].T.dot(delta_j) * act_f(self.activation, 1)(net[0])  # vector delta_i
        grad_list.append(np.outer(pre_delta, o[0]))  # compute and append the derivatives wrt weights
        grad_bias_list.append(pre_delta)  # append the derivatives wrt biases
        return pre_delta, grad_list, grad_bias_list

    def gradient(self, input, target):
        """ Takes input and target from one example <x,y>
        and updates the gradient in self.Gradient_list and
        self.Gradient_bias_list
        """
        output, net_matrix = self.saved_feedforward(input)
        grad_list, grad_bias_list = self.backprop(input, target, output, net_matrix, [], [])[1:]
        for item in [grad_list, grad_bias_list]:
            item.reverse()
        return np.array([grad_list, grad_bias_list], dtype=object)

    def training(self, data_train, epochs, mb, eta):
        eta = eta / mb
        for epoch in range(epochs):
            for i in range(math.ceil(len(data_train) / mb)):
                delta_w = [np.zeros((y, x)) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
                delta_b = [np.zeros(x) for x in self.sizes[1:]]
                gradients = np.array([delta_w, delta_b], dtype=object)
                for j in range(mb * i, mb * (i + 1)):
                    if j == len(data_train):
                        return
                    gradients = np.add(gradients, self.gradient(data_train[j][0], data_train[j][1]))
                self.hidden_layers = np.add(gradients[0] * eta, self.hidden_layers)
                self.biases = np.add(gradients[1] * eta, self.biases)

#aggiungere scelta della funzione d'attivazione
#cross validation

#misure di assesment
#   plot della learning curve

#===Implementazioni rimanenti===
#   eta decay?
#

#===Iperparametri da cercare===
#   lambda
#   eta
#   alpha (Momentum)
#   a (coefficente della sigmoide)
#   *funzione d'attivazione
#   batch VS online


def cross_validation (model, dataset, K, hyperparameters):
    fold_size = len(dataset) // K
    best_pair = (-1,0)

    for theta in hyperparameters_grid:
        for i in range(K):
            model.training(dataset[:i*fold_size]+dataset[(i+1)*fold_size:], theta) #training == select h(theta,\Bar{D^k})
            R_emp += model.emp_error(dataset[i*fold_size:(i+1)*fold_size]) #estimate R through the empirical error on the current D^k
                                                                           #we sum because we are not interested in the single R_emp,
                                                                           #we need them all to do an average over the folds
        err = R_emp/K #estimate R on the hyperparameter theta through an average over the folds
        #if 1st iteration then update
        if best_pair[0] == -1:
            best_pair = (err, theta)
        #if not 1st iteration update only if the error improved
        else:
            if err < best_pair[0]:
                best_pair = (err, theta)
        err = 0   #reset for next iteration
        R_emp = 0   #reset for next iteration

    best_theta = best_pair[1]

    model.training(dataset, best_theta)   #select best hypothesis by retraining the model on whole dataset with the best hyperprameters   
