import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import random
import itertools
import time


def encode_dataset(dataset, start_input_column=2, total_columns=6, target_column=1):
    dd={}
    i=0
    for column in range(total_columns):
        m = max(set(dataset.iloc[:, start_input_column+column]))
        dd[i] = m 
        i+=1
    input_range=np.sum([v for v in dd.values()])
    DS=[]
    for line in range(len(dataset)):
        input_=np.zeros(input_range)
        l=list(dataset.iloc[line, start_input_column:total_columns+start_input_column])
        target=dataset.iloc[line, target_column]
        i=0
        index=0
        for value in l:
            input_[index+value-1]=1
            index+=dd[i]
            i+=1
        example=input_,target
        DS.append(example)
    return DS



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

    def __init__(self, sizes, activation, weights_sigma=1):
        """ sizes is a list of dimensions (number of units for each layer)
            activation is a string (sigmoid/relu) related to activation function
            used in the hidden units"""
        self.seed = random.randrange(0, (2**32)-1)
        self.sizes = sizes
        self.weights_sigma=weights_sigma
        self.activation = activation
        self.n_input = sizes[0]  # number of input units
        self.n_output = sizes[-1]  # number of output units
        np.random.seed(self.seed)
        self.hidden_layers = np.array([self.weights_sigma*np.random.randn(y, x) for (y, x) in zip(self.sizes[1:], self.sizes[:-1])], dtype=object)  # list of weight matrices for hidden layers
        np.random.seed(self.seed)
        self.biases = np.array([self.weights_sigma*np.random.random(x) for x in self.sizes[1:]], dtype=object)

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
        """Compute the derivatives of the -MSE cost function wrt weights and biases """
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

    def training(self, data_train, epochs=100, mb=1, eta=1, theta={}, momentum=False, alpha_mom=1, regularization=False, lambda_reg=0.001):
        if "eta" in theta:
            eta=theta["eta"]
        if "alpha_mom" in theta:
            alpha_mom=theta["alpha_mom"]
            momentum=True
        if "lambda_reg" in theta:
            lambda_reg=theta["lambda_reg"]
            regularization=True
        eta = eta / mb
        if momentum:
            old_gradients=np.array([[np.zeros((y, x)) for y, x in zip(self.sizes[1:], self.sizes[:-1])], [np.zeros(x) for x in self.sizes[1:]]])
        for epoch in range(epochs):
            random.shuffle(data_train) # shuffle dataset between epochs
            for i in range(math.ceil(len(data_train) / mb)):
                delta_w = [np.zeros((y, x)) for y, x in zip(self.sizes[1:], self.sizes[:-1])]
                delta_b = [np.zeros(x) for x in self.sizes[1:]]
                gradients = np.array([delta_w, delta_b], dtype=object)
                for j in range(mb * i, mb * (i + 1)):
                    if j < len(data_train):
                      gradients = np.add(gradients, self.gradient(data_train[j][0], data_train[j][1]))

                ## added momentum and regularization
                if momentum:
                    gradients = np.add(gradients, old_gradients)
                    old_gradients= [alpha_mom*gradient for gradient in gradients]
                if regularization:
                    lambda_W = [-lambda_reg*layer for layer in self.hidden_layers]
                    lambda_b = [-lambda_reg*bias for bias in self.biases]
                    self.hidden_layers = np.add(lambda_W, self.hidden_layers)
                    self.biases = np.add(lambda_b, self.biases)
                #print(self.cost_function(data_train))
                self.hidden_layers = np.add(gradients[0] * eta, self.hidden_layers)
                self.biases = np.add(gradients[1] * eta, self.biases)

    ## R_emp measures:

    # Misclassification Rate = # incorrect predictions / # total predictions
    def MR(self, data):
        pass

    # Mean square error
    def MSE(self, data):
        E_sum=0
        for x in data:
            E_sum+=(self.feedforward(x[0]) - x[1])**2
        return E_sum/len(data)

    # Mean absolute error
    def cost_function(self, data):
        c=0
        for x in data:
            c += abs(x[1]-self.feedforward(x[0]))
        return c/len(data)
    
    def errors_count(self, data):
        errors = 0
        for x in data:
            if round(float(self.feedforward(x[0])),0) != x[1]:
                errors += 1
        return errors/len(data)
    
    
    def accuracy(self, data):
        correct = 0
        for x in data:
            if round(float(self.feedforward(x[0])),0) == x[1]:
                correct += 1
        return correct/len(data)*100

    
    def reset_weights(self, same_init_seed=False):
        if same_init_seed:
            np.random.seed(self.seed)
        self.hidden_layers = np.array([self.weights_sigma*np.random.randn(y, x) for (y, x) in zip(self.sizes[1:], self.sizes[:-1])], dtype=object)  # list of weight matrices for hidden layers
        if same_init_seed:
            np.random.seed(self.seed)
        self.biases = np.array([self.weights_sigma*np.random.random(x) for x in self.sizes[1:]], dtype=object)

    def learning_curve(self, ax, Cost_training):
        ax.plot(Cost_training)
        ##add ax.plot(MSE_validation)

###
"""
    def search_eta(self,eta_list, data_train, epochs=300, eta_iteration=5):
        l=int(np.sqrt(len(eta_list)))
        if np.sqrt(len(eta_list))>l:
            l=l+1
        ##find grid dimension from eta array lenght
        fig, axarr = plt.subplots(l, l)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Learning Curves", fontsize=18, y=0.95)
        #search eta #parameter
        for e, ax in zip(eta_list, axarr.ravel()):
            for i in range(eta_iteration):
                eta = self.training(data_train, epochs=epochs, eta=e)
                self.learning_curve(ax)
                self.reset()
            ax.set_title("eta/l = {}".format(eta))
            ax.set_xlabel("epochs")
            ax.set_ylabel("MSE")
        plt.show()
"""

#   cost function regression VS classification

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

#def grid_search(hyperparameters):
#    """hyperparameters is a dictionary where keys are the hyperparameters and 
#    values lists of their values e.g. {"eta": [0.1, 0.01, 0.001], "lambda_reg":[0.1, 0.01, 0.001]}"""
#    ## use np.arange to get an array of float values
#    keys, values = zip(*hyperparameters.items())
#    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
#    return permutations_dicts
#    
#def random_grid(hyperparameters, samples=10):
#    return [dict(zip(hyperparameters.keys(), [random.uniform(min(v), max(v)) for v in hyperparameters.values()])) for i in range(5)]

def cross_validation(model, dataset, hyperparameters, K=5, mb=1, search_type="grid", samples=10, convergence_sample=30, error_threshold=0.0001):
    fold_size = len(dataset) // K
    best_pair = (-1,0)
    if search_type == "grid":
        values = [np.linspace(min(v), max(v), num=samples) for v in hyperparameters.values()]
        hyperparameters_grid = [dict(zip(hyperparameters.keys(), v)) for v in itertools.product(*values)]
    if search_type == "random":
        hyperparameters_grid = [dict(zip(hyperparameters.keys(), [random.uniform(min(v), max(v)) for v in hyperparameters.values()])) for i in range(samples)]
    for theta in hyperparameters_grid:
        R_emp=0 
        err=0 
        TR_errs=0
        for i in range(K):
            model.reset_weights()
            TR_error=model.MSE(dataset[:i*fold_size]+dataset[(i+1)*fold_size:])
            e=0
            error_percentage=100
            TR_error_old=[]
            while error_percentage > error_threshold:
                model.training(dataset[:i*fold_size]+dataset[(i+1)*fold_size:], theta=theta, mb=mb, epochs=1) #training == select h(theta,\Bar{D^k})
                TR_error_old.append(TR_error)
                TR_error = model.MSE(dataset[:i*fold_size]+dataset[(i+1)*fold_size:])
                if e > convergence_sample:
                    error_percentage = (TR_error-np.mean(TR_error_old))/TR_error
                    TR_error_old.pop(0)
                e+=1
            R_emp += model.errors_count(dataset[i*fold_size:(i+1)*fold_size]) #estimate R through the empirical error on the current D^k
                                                                           #we sum because we are not interested in the single R_emp,
                                                                           #we need them all to do an average over the folds
            TR_errs += TR_error
        err = R_emp/K #estimate R on the hyperparameter theta through an average over the folds
        mean_TR_errs = TR_errs/K #used to match the same level of fitting for the final model
        #if 1st iteration then update
        if best_pair[0] == -1:
            best_pair = (err, theta, mean_TR_errs)
        #if not 1st iteration update only if the error improved
        else:
            if err < best_pair[0]:
                best_pair = (err, theta, mean_TR_errs)
        #err = 0   #reset for next iteration
        #R_emp = 0   #reset for next iteration
        print("theta n°", hyperparameters_grid.index(theta), time.ctime(), mean_TR_errs)
    best_theta = best_pair[1]
    TR_threshold = best_pair[2]
    print(best_pair)
    return best_theta, TR_threshold
    #model.training(dataset, theta=best_theta)   #select best hypothesis by retraining the model on whole dataset with the best hyperprameters   
    
def final_training(model, data_train, test_set, theta, TR_threshold, mb=1, learning_curve=False, mse=True, accuracy=True, stopping_criteria="convergence", convergence_sample=30, error_threshold=0.0001, escaping_epochs=300):
    model.reset_weights()
    TR_accuracy = model.accuracy(data_train)
    TR_mse = model.MSE(data_train)
    if learning_curve:
        TR_cost=[]
        TS_cost=[]
        TR_acc=[]
        TS_acc=[]
    e=0
    if stopping_criteria=="threshold":
        while TR_mse > TR_threshold:
            model.training(data_train, mb=mb, epochs=1, theta=theta)
            TR_accuracy = model.accuracy(data_train)
            TR_mse = model.MSE(data_train)
            if learning_curve:
                TR_cost.append(TR_mse)
                TR_acc.append(TR_accuracy)
            TS_accuracy = model.accuracy(test_set)
            TS_mse = model.MSE(test_set)
            if learning_curve:
                TS_acc.append(TS_accuracy)
                TS_cost.append(TS_mse)
            e+=1
            if e > escaping_epochs:
                break
    if stopping_criteria=="convergence":
        TR_mse=model.MSE(data_train)
        e=0
        error_percentage=100
        TR_error_old=[]
        while error_percentage > error_threshold:
            model.training(data_train, theta=theta, mb=mb, epochs=1) #training == select h(theta,\Bar{D^k})
            TR_error_old.append(TR_mse)
            TR_accuracy = model.accuracy(data_train)
            TR_mse = model.MSE(data_train)
            if learning_curve:
                TR_cost.append(TR_mse)
                TR_acc.append(TR_accuracy)
            TS_accuracy = model.accuracy(test_set)
            TS_mse = model.MSE(test_set)
            if learning_curve:
                TS_acc.append(TS_accuracy)
                TS_cost.append(TS_mse)
            if e > convergence_sample:
                error_percentage = (TR_mse-np.mean(TR_error_old))/TR_mse
                TR_error_old.pop(0)
            e+=1
            if e > escaping_epochs:
                break
    print("TS Accuracy: ", TS_accuracy, "TR Accuracy: ", TR_accuracy, "epochs: ", e, "TS MSE: ", TS_mse, "TR MSE: ", TR_mse)
    ##
    if learning_curve:
        N=0
        if accuracy:
            N+=1
        if mse:
            N+=1
        fig, axarr = plt.subplots(1, N)
        plt.subplots_adjust(hspace=0.5)
        fig.suptitle("Learning Curves", fontsize=18, y=0.95)
        measure_list=[]
        if accuracy:
            c = (TR_acc,TS_acc, "Accuracy")
            measure_list.append(c)
        if mse:
            c = (TR_cost,TS_cost, "Cost Function")
            measure_list.append(c)
        for measures, ax in zip(measure_list, axarr.ravel()):
            ax.plot(measures[0], label="Training Set")
            ax.plot(measures[1], label="Test Set")
            ax.set_xlabel("epochs")
            ax.set_ylabel(measures[2])
            ax.legend()
        plt.show()