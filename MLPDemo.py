from MLP import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

monk1= pd.read_csv(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_fixed.csv")
data_train = encode_dataset(monk1, start_input_column=2, total_columns=6, target_column=1)
#data_train

test = pd.read_csv(r"C:\Users\Vittorio\Documents\UNIPI\Machine Learning\Progetto_ML\main_02\monks1_test.csv")
test_set = encode_dataset(test, start_input_column=1, total_columns=6, target_column=0)
#test_set

NN=MLP([17,3,1], "sigmoid", weights_sigma=0.75)
hyperparameters={"eta": [1,0.1], "alpha_mom": [1,0.1], "lambda_reg": [1,0.001]} #"weights_sigma":[0.15,1]
#best_theta, TR_treshold = cross_validation(NN, data_train, hyperparameters, K=4, search_type="grid", samples=4, mb=len(data_train), convergence_sample=500)
best_theta={'eta': 1, 'alpha_mom': 0.5, 'lambda_reg': 0.001}
TR_treshold=100
final_training(NN, data_train[:-50], data_train[-50:], best_theta, TR_treshold, mb=len(data_train[:-50]), learning_curve=True, accuracy=True, mse=True, convergence_sample=1000, stopping_criteria="convergence", error_threshold=0.00000001, escaping_epochs=1000)
