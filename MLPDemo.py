from MLP import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

monk1= pd.read_csv(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_fixed.csv")
inputs = monk1[["a1", "a2", "a3", "a4", "a5", "a6"]]#pd.DataFrame(monk1.columns !='6')
labels = monk1["class"]#pd.DataFrame(monk1, columns=['6'])
inputs = inputs.values.tolist()
labels = labels.values.tolist()
data_train = [[input, label] for (input,label) in zip(inputs,labels)]


#with open(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_test.txt", "r") as f:
#    with open(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_test.csv", "a") as w:
#        for line in f.readlines():
#            w.write(line.replace(" ", ",")[1:])

test = pd.read_csv(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_test.csv")
inputs_test = test[["1.1","1.2","1.3","1.4","1.5","1.6"]]#pd.DataFrame(test, columns=['a1','a2','a3','a4','a5','a6'])
inputs_test = inputs_test.values.tolist()
labels_test = test["1"]#pd.DataFrame(test, columns=['class'])
labels_test = labels_test.values.tolist()
test_set = [[input, label] for (input,label) in zip(inputs_test,labels_test)]


NN=MLP([6,3,1], "sigmoid")
hyperparameters={"eta": [1,0.001], "alpha_mom": [1,0.001], "lambda_reg": [1,0.001]}
best_theta, TR_treshold = cross_validation(NN, data_train, hyperparameters, K=4, search_type="random", samples=50)
final_training(NN, data_train, test_set, best_theta, TR_treshold, mb=1, learning_curve=False)
