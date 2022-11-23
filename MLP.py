import numpy as np
import pandas as pd
import matplotlib as plt


def sigmoid(x, a=1):
    return 1/(1 + np.exp(-(a*x)))

def sigmoid_derivative(x,a=1):
    return a*sigmoid(x,a)*(1-sigmoid(x,a))

def E_p(o_k, d):
    return (o_k-d)**2

def E(example_list):
    #example_list in shape [(x_p,d_p),...]
    E_sum=0
    for example in example_list:
        E_sum+=E_p(example[0],example[1])
    return E_sum

class MLP:

    def __init__(self, sizes):
        # sizes is a list of dimensions (number of units for each layer)
        self.sizes = sizes
        self.n_input = sizes[0]  # number of input units
        self.n_output = sizes[-1]  # number of output units
        self.hidden_layers = [np.random.randn(y, x) for (y, x) in
                              zip(self.sizes[1:], self.sizes[:-1])]  # list of weight matrices for hidden layers
        self.biases = [np.random.random(x) for x in self.sizes[1:]]

    def feedforward(self, input):
        for (W, b) in zip(self.hidden_layers[:-1], self.biases[:-1]):
            input = sigmoid(W @ input + b)
        net = self.hidden_layers[-1].dot(input) + self.biases[-1]  # compute the net for the output layer
        output = sigmoid(net)  # compute the output for the output layer
        return output

    def saved_feedforward(self, input):
        output = [input]
        net_matrix = []
        for i, (W, b) in enumerate(zip(self.hidden_layers[:-1], self.biases[:-1])):
            net = W.dot(input) + b  # compute the net for a hidden layer
            o = sigmoid(net)  # compute the output for a hidden layer
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
            delta_k = (np.array(target) - np.array(o[1])) * np.array(sigmoid_derivative(net[0]))  # compute the final delta (delta_L)
            derivatives = np.outer(delta_k, o[0])  # compute the derivatives wrt last weights, here o[0]=o_(L-1)
            grad_list.append(derivatives)
            grad_bias_list.append(delta_k) # append derivatives wrt biases that coincides with delta_L
            return delta_k, grad_list, grad_bias_list
        i += 1
        delta_j = self.backprop(input, target, o[1:], net[1:], grad_list, grad_bias_list, i)[0]  # recursive call (delta_(i+1))
        pre_delta = np.array(self.hidden_layers[i]).T.dot(delta_j) * np.array(sigmoid_derivative(net[0]))  # vector delta_i
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
        return np.array([grad_list, grad_bias_list])

    def training(self, data_train, epochs, mb, eta):
        eta = eta / mb
        for epoch in range(epochs):
            for i in range(np.ceil(len(data_train) / mb)):
                delta_w = [np.zeros((y, x)) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
                delta_b = [np.zeros(x) for x in self.sizes[1:]]
                gradients = np.array([delta_w, delta_b])
                for j in range(mb * i, mb * (i + 1)):
                    if j == len(data_train):
                        return
                    gradients = np.add(gradients, self.gradient(data_train[j][0], data_train[j][1]))
                self.hidden_layers = np.add(gradients[0] * eta, self.hidden_layers)
                self.biases = np.add(gradients[1] * eta, self.biases)
                
                
#   aggiungere scelta della funzione d'attivazione
#   cross validation

#   misure di assesment
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
