import numpy as np
import math
import random
from numpy.random import default_rng
import activation_functions
rng = default_rng()

   
class MLP:

    def __init__(self, sizes, activation, weights_std=1/3, task = 'regression',
                 xavier = True, a=1):
        """ sizes is a list of dimensions (number of units for each layer)
            activation is a string (sigmoid/relu) related to activation function
            used in the hidden units, #n of hidden units can be an hyperparameter"""
        self.sizes = sizes
        self.weights_std = weights_std
        self.activation = activation
        self.a = a
        self.task = task
        self.xavier = xavier
        self.n_input = sizes[0]  # number of input units
        self.n_output = sizes[-1]  # number of output units
        if xavier:
          self.hidden_layers = np.array([rng.uniform(-(6/(y+x))**(0.5),
                                                     (6/(y+x))**(0.5),
                                                     (y,x)).tolist()
                                         for (y,x) in 
                                         zip(self.sizes[1:], self.sizes[:-1])],
                                         dtype = object)
        else:
          self.hidden_layers = np.array([(rng.normal(0,weights_std,(y,x))).tolist()
                                      for (y, x) in
                                      zip(self.sizes[1:], self.sizes[:-1])], 
                                      dtype=object)
        # list of weight matrices for hidden layers
        self.biases = np.array([np.zeros(x)
                                for x in self.sizes[1:]], 
                                dtype=object)
        # list of biases vector

    def reset_weights(self):
        """Method that resets the weights according to 
        the initialization strategy of the MLP object"""
        #resets the biases
        self.biases = np.array([np.zeros(x)
                                for x in self.sizes[1:]], 
                                dtype=object)
        #resets the layer weigths
        if self.xavier:
          self.hidden_layers = np.array([rng.uniform(-(6/(y+x))**(0.5),
                                                     (6/(y+x))**(0.5),
                                                     (y,x)).tolist()
                                         for (y,x) in 
                                         zip(self.sizes[1:], self.sizes[:-1])],
                                         dtype = object)
        else:
          self.hidden_layers = np.array([(rng.normal(0,self.weights_std,(y,x))).tolist()
                                      for (y, x) in
                                      zip(self.sizes[1:], self.sizes[:-1])], 
                                      dtype=object)


    def feedforward(self, input):
        """Method that propagates the input trough the neural network
        and returns the output of the model"""
        for (W, b) in zip(self.hidden_layers[:-1], self.biases[:-1]):
            input = activation_functions.act_f(self.activation, 0)(W @ input + b, a=self.a)
        net = (self.hidden_layers[-1])@input + self.biases[-1]
        # compute the net for the output layer
        if self.task == 'classification':
            output = activation_functions.sigmoid(net, a=self.a)
        elif self.task == 'regression':
            output = net
        # compute the output for the output layer
        return output

    def saved_feedforward(self, input):
        """Feedforward method that saves the nets and the outputs 
        for each layer to use them for computing the gradients"""
        output = [input]
        net_matrix = []
        for i, (W, b) in enumerate(zip(self.hidden_layers[:-1],
                                       self.biases[:-1])):
            net = W@input + b  # compute the net for a hidden layer
            out = activation_functions.act_f(self.activation, 0)(net, a=self.a)  # compute the output
            input = out  # assigning inputs
            output.append(out)  # save outputs
            net_matrix.append(net)  # save nets
        net = self.hidden_layers[-1]@input + self.biases[-1]
        # compute the net for the output layer*/
        if self.task == 'classification':
            output.append(activation_functions.sigmoid(net, a=self.a))
        elif self.task == 'regression':
            output.append(net)
        # compute the output for the output layer
        net_matrix.append(net)
        return output, net_matrix

    def backprop(self, input, target, o, net, grad_list, grad_bias_list, i=0):
        """Compute the derivatives of
        the -MSE cost function wrt weights and biases """
        if i == (len(self.hidden_layers) - 1):  # last layer
            if self.task == 'classification':
                delta_k = (target - o[1]) * activation_functions.sigmoid_derivative(net[0], a=self.a)
            elif self.task == 'regression':
                delta_k = (target-o[1])
            # compute the final delta (delta_L)
            derivatives = np.outer(delta_k, o[0])
            # compute the derivatives wrt last weights, here o[0]=o_(L-1)
            grad_list.append(derivatives)
            grad_bias_list.append(delta_k)
            # append derivatives wrt biases that coincides with delta_L
            return delta_k, grad_list, grad_bias_list
        i += 1
        delta_j = self.backprop(input,target, o[1:], net[1:], grad_list,
                                grad_bias_list, i)[0]
                                # recursive call (delta_(i+1))
        pre_delta = np.transpose(self.hidden_layers[i])@(delta_j) \
                    *activation_functions.act_f(self.activation, 1)(net[0], a=self.a)
                    # vector delta_i
        grad_list.append(np.outer(pre_delta, o[0]))  
        # compute and append the derivatives wrt weights
        grad_bias_list.append(pre_delta)  # append the derivatives wrt biases
        return pre_delta, grad_list, grad_bias_list

    def gradient(self, input, target):
        """ Takes input and target from one example <x,y>
        and updates the gradient in self.Gradient_list and
        self.Gradient_bias_list
        """
        output, net_matrix = self.saved_feedforward(input)
        grad_list, grad_bias_list = self.backprop(input, target, output,
                                                  net_matrix, [], [])[1:]
        #reverse the results of backpropagation method to match the hidden layers
        for item in [grad_list, grad_bias_list]: 
            item.reverse()
        return np.array([grad_list, grad_bias_list], dtype=object)

    def training(
          self, 
          data_train,
          epochs=300,
          mb=124,
          eta=1,
          theta={},
          lambda_reg = [],
          test_set=[],
          eta_decay=True,
          tau = 30
        ):
        momentum = False
        regularization = False
        prev_train_cost = 0
        prev_train_acc = 0
        count_stall = 0
        count_stall2 = 0
        train_cost = []
        train_acc = []
        test_cost = []
        test_acc = []
        # take possible hyperparameters
        if "eta" in theta:
            eta = theta["eta"]
        if "alpha_mom" in theta:
            alpha_mom = theta["alpha_mom"]
            momentum=True
        if "lambda_reg" in theta:
            lambda_reg = theta["lambda_reg"]
            regularization = True
        if "epochs" in theta:
            epochs = theta["epochs"]
        if "mb" in theta:
            mb = theta["mb"]
        if "eta_decay" in theta:
            eta_decay = theta["eta_decay"]
        if "tau" in theta:
            tau = theta["tau"]
	# option for momentum
        if momentum:
            old_gradients=np.array([[np.zeros((y, x)) 
                                     for y, x in 
                                     zip(self.sizes[1:], self.sizes[:-1])],
                                    [np.zeros(x) for x in self.sizes[1:]]],
                                    dtype=object)
	# option for eta decay
        eta = eta / mb
        if eta_decay:
            eta_0 = eta
            eta_tau = eta_0/tau
	# begin the training by iterating through the number of epochs.
        for epoch in range(epochs):
            if eta_decay == True and epoch-40 <= tau and epoch>=40:
                eta = (1-(epoch-40)/tau)*eta_0 + ((epoch-40)/tau)*eta_tau
            #shuffles dataset to avoid biases due to order
            data_train = random.sample(data_train,len(data_train))
            #iterate over each minibatch
            for i in range(math.ceil(len(data_train) / mb)):
                delta_w = np.array([np.zeros((y, x))
                                    for y, x in
                                    zip(self.sizes[1:], self.sizes[:-1])],
                                   dtype=object)
                delta_b = np.array([np.zeros(x) for x in self.sizes[1:]],
                                   dtype=object)
                gradients = np.array([delta_w, delta_b], dtype=object)
                #iterate over each example in the current minibatch
                for j in range(mb * i, mb * (i + 1)):
                    if j < len(data_train): #gradient over the i-th batch
                      gradients = np.add(gradients,
                                         self.gradient(data_train[j][0],
                                                       data_train[j][1]))
                #regularization condition: adds lambda*W component
                if regularization:
                    lambda_w = [-lambda_reg*np.array(layer)
                                for layer in self.hidden_layers]
                    lambda_b = [-lambda_reg*bias for
                                bias in self.biases]
                    gradients = np.add(np.array([lambda_w, lambda_b],dtype=object),
                                       gradients)
                #momentum condition: adds alpha*old_gradient component
                if momentum:
                    gradients = np.add(gradients, old_gradients)
                    old_gradients = [alpha_mom*gradient 
                                     for gradient in gradients]
                self.hidden_layers = np.add(gradients[0] * eta,
                                            self.hidden_layers) #update the weights
                self.biases = np.add(gradients[1] * eta, self.biases) #update the biases
            #compute cost fuctions for each epoch
            train_cost += [self.error_function(data_train,
                                               lambda_reg = lambda_reg)]
            if self.task == 'classification':
                train_acc += [self.error_function(data_train, accuracy_measure=True)]
            #given a test_set returns an assesment
            if test_set != []:
                test_cost +=[self.error_function(test_set,
                                                 lambda_reg = lambda_reg)]
                if self.task == 'classification':
                    test_acc += [self.error_function(test_set, accuracy_measure=True)]
            
            #==========stop conditions==========
            if self.task == 'regression':
                #convergence condition for the error_function (MEE for regression)
                if (self.error_function(data_train, lambda_reg = lambda_reg) >= 
                    (1-1/250)*prev_train_cost):
                    count_stall += 1
                else:
                    count_stall = 0
                prev_train_cost = self.error_function(data_train, lambda_reg = lambda_reg)
            elif self.task == 'classification':
                #convergence condition for the error_function
                #(MSE for classification) in regularized and unregularized cases
                if regularization:
                    if (self.MSE_reg(data_train, lambda_reg) >= 
                        prev_train_cost-(1/3000)*prev_train_cost):
                        count_stall += 1
                    else:
                        count_stall = 0
                    prev_train_cost = self.MSE_reg(data_train, lambda_reg)
                else:
                    if (self.MSE(data_train) >= 
                        prev_train_cost-(1/3000)*prev_train_cost):
                        count_stall += 1
                    else:
                        count_stall = 0
                    prev_train_cost = self.MSE(data_train)
                #convergence condition for the accuracy measure
                if self.error_function(data_train, accuracy_measure=True) <= prev_train_acc:
                    count_stall2 += 1
                else:
                    count_stall2 = 0
                prev_train_acc = max(self.error_function(data_train, accuracy_measure=True),
                                 prev_train_acc)
            #escaping conditions given number of consecutive stalls
            if count_stall == 50 or (prev_train_acc > 94 and count_stall2 == 80):
                # print("Stall over 90:", epoch-49)
                break
            elif count_stall2 == 150:
                # print("Stall:", epoch-149)
                break
        return train_cost, test_cost, train_acc, test_acc

    def MSE(self, data):
        """Mean Squared Error given a set of data"""
        err_sum = 0
        for x in data:
            err_sum += (self.feedforward(x[0]) - x[1])**2
        return err_sum/len(data)
    
    def MSE_reg(self, data, lambda_reg):
        """L^2 regularization of MSE given a set of data"""
        norm = 0
        for W in self.hidden_layers:
            norm += np.linalg.norm(W)**2
        for b in self.biases:
            norm += np.linalg.norm(b)**2
        return self.MSE(data) + lambda_reg*np.array(norm)
    
    def errors_count(self, data):
        """Errors count given a set of data"""
        errors = 0
        for x in data:
            if round(float(self.feedforward(x[0])),0) != x[1]:
                errors += 1
        return errors/len(data)

    def accuracy(self, data):
        """Accuracy measure given a set of data"""
        correct = 0
        for x in data:
            if round(float(self.feedforward(x[0])),0) == x[1]:
                correct += 1
        return correct/len(data)*100
    
    def MEE(self, data):
        """Mean Euclidean Error given a set of data"""
        err_sum = 0
        for x in data:
            err_sum += np.linalg.norm(self.feedforward(x[0]) - x[1])
        return err_sum/len(data)

    def MEE_reg(self, data, lambda_reg):
        """L^2 regularization of MEE given a set of data"""
        norm = 0
        for W in self.hidden_layers:
            norm += np.linalg.norm(W)**2
        for b in self.biases:
            norm += np.linalg.norm(b)**2
        return self.MEE(data) + lambda_reg*norm

    def error_function(self, data, lambda_reg = None, accuracy_measure = False):
        """Ensemble method of the different error functions"""
        if lambda_reg != None:
            regularization = True
        else:
            regularization = False
        if self.task == 'classification':
            if accuracy_measure:
                return self.accuracy(data)
            if regularization:
                return self.MSE_reg(data, lambda_reg)
            else:
                return self.MSE(data)
        elif self.task == 'regression':
            if accuracy_measure:
                pass
            if regularization:
                return self.MEE_reg(data, lambda_reg)
            else:
                return self.MEE(data)