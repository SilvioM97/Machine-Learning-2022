import numpy as np
import time
import random
from numpy.random import default_rng
import itertools
import concurrent.futures
import matplotlib.pyplot as plt
import mlp

rng = default_rng()

def cross_validation(dataset,
                     hyperparameters,
                     K, cross_times,
                     task="regression",
                     max_workers=40, 
                     random_size = None):
    """Given a set of hyperparameters builds a grid
    (or a sample) and performs <cross_times>
    time a K-fold Validation and returns the 5 best 
    combination of hyperparameters"""
    fold_size = len(dataset) // K 
    values = [v for v in hyperparameters.values()] 
    hyperparameters_grid = [dict(zip(hyperparameters.keys(), v))
                            for v in itertools.product(*values)] 
    best_hyps = [(97, 0) for i in range(5)] #initialize the 5 best hyperparameter combinations
    if random_size != None:
        hyperparameters_grid = rng.choice(hyperparameters_grid, random_size) #samples the grid
    for theta in hyperparameters_grid:
        train_accuracy = 0
        val_errors = 0
        list_train_accuracy = []
        list_val_errors = []
        model = mlp.MLP(theta["architecture"],
                    theta["activation"],
                    xavier = theta["xavier"],
                    a = theta["a"],
                    task = task)
        for j in range(cross_times): #iterate <cross_times> time over each combination of hyperparameters
            rng.shuffle(dataset) #shuffle the dataset to avoid biases due to order of the examples
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_fold = {executor.submit(train_and_evaluate,
                                                  model,
                                                  theta,
                                                  dataset,
                                                  i,
                                                  fold_size): i for i in range(K)}
                                                  #begin the training on each fold and evaluates the model
                for future in concurrent.futures.as_completed(future_to_fold): #iterate over each fold
                    train_acc, val_err = future.result() #results for a fold
                    train_accuracy += train_acc
                    val_errors += val_err
                    list_train_accuracy += [train_acc] #list_train_acc
                    list_val_errors += [val_err] #list_val_err
        #validation estimation for a hyperparameter combination
        mse = round((val_errors/(K*cross_times))**2 + np.var(list_val_errors),4)
        #select the best 5 theta given the previous result
        for i, hyp in enumerate(best_hyps):
            if mse < hyp[0]:
                old_shift = best_hyps[i:-1]
                best_hyps[i] = (mse,
                                round(val_errors/(K*cross_times),2),
                                round(np.std(list_val_errors),2),
                                round(float(train_accuracy)/(K*cross_times),2),
                                theta)
                best_hyps = best_hyps[:i+1] + old_shift
                break
        print(theta,
              time.ctime())
        #print the cost functions
        if task == 'regression':
                print("MEE:", round(val_errors/(K*cross_times),4),
                    "MEE_std:",round(np.std(list_val_errors),4),
                    "Training info:", round(float(train_accuracy)/(K*cross_times),4),
                    round(np.std(list_train_accuracy),4))
        elif task == 'classification':
                print("0-1 Loss on VL:", round(val_errors/(K*cross_times),4),
                    "0-1 Loss_std:",round(np.std(list_val_errors),4),
                    "Training info:", round(float(train_accuracy)/(K*cross_times),2),
                    round(np.std(list_train_accuracy),2))
    #print the 5 best theta
    for bh in best_hyps:
        print(bh)
    return best_hyps

def train_and_evaluate(model, theta, dataset, i, fold_size):
    """Trains a model given a theta 
    (combination of hyperparameters) and
    returns the train and validation
    cost functions depending on the training task"""
    model.reset_weights()
    model.training(dataset[:i*fold_size]+dataset[(i+1)*fold_size:],
                   theta = theta) # train on the ith fold
    if model.task == 'regression':
            train_acc = model.error_function(dataset[:i*fold_size]+dataset[(i+1)*fold_size:],
                                            theta["lambda_reg"]
                                            ) # train_acc on the ith fold
            val_err = model.error_function(dataset[i*fold_size:(i+1)*fold_size]) 
            # val_err on the validation fold
    elif model.task == 'classification':
            train_acc = model.accuracy(dataset[:i*fold_size]+dataset[(i+1)*fold_size:])
            # train_acc on the ith fold
            val_err = model.errors_count(dataset[i*fold_size:(i+1)*fold_size])
            # val_err on the validation fold

    return train_acc, val_err



def ensemble(training_set, best_hyps, test_set, task='regression'):
  """Takes an input set and predicts the labels by ensembling 
     first the best hyperparameters found during cross validation
     (by raw result, before transforming the result into a class 
     label in the case of classification) and then the results 
     obtained by 5 different weights initializations (equally weighted vote)"""
  vote = [0 for i in range(len(test_set)+len(training_set))] #sums of the outputs of different models
  results = [] #accuracy and cost function (train and test) for each model

  for j in range(5): #for each hyperparameter we take 5 different weights initializations
      output = [0.0 for i in range(len(test_set)+len(training_set))] 
      for theta in best_hyps: #best hyperparameters combinations
          model = mlp.MLP(theta["architecture"],
                      theta["activation"],
                      xavier = theta["xavier"],
                      a = theta["a"],
                      task=task) #new model
          results.append(model.training(training_set, theta = theta, test_set=test_set)) #accuracies and costs (train and test)
          for i, input in enumerate(np.concatenate((training_set, test_set))):
              output[i] += model.feedforward(input[0]) #compute the outputs for the current model
      for i in range(len(test_set)+len(training_set)): 
          output[i] /= len(best_hyps) #average of the best hyperparameters' models
          if task == 'classification':
              vote[i] += round(float(output[i])) #vote of the 5 different classifiers
          elif task == 'regression':
              vote[i] += output[i]  #sum for each averaged model the outputs for 5 weights

  if task == 'classification':
      output = [round(float(x/5)) for x in vote] #round transform the average of the 5 weights initializations runs into a class
  elif task == 'regression':
      output = [x/5 for x in vote] #average of the 5 weights initializations runs

  return output, results



def tests (training_set, test_set, hyps, task='regression'):
  """Given a set of hyperparameter combinations, a training set
  and a test set, trains an ensemble with the specified hyperparameters
  and returns plots and accuracies both on training set and test set"""

  out, results = ensemble(training_set, hyps, test_set, task=task) #run tests on test set
  out_tr = out[:len(training_set)]
  out_ts = out[len(training_set):]

  if task == 'regression':
      for i in range(len(results)):
          results[i] = (results[i][0], results[i][1])

  max_length = max([len(results[i][0]) for i in range(len(hyps)*5)])
  for i in range(len(results)):
      for j in range(len(results[i])):
          list_to_expand = results[i][j]
          expand_by = max_length - len(results[i][j])
          results[i][j].extend([results[i][j][-1]] * expand_by)
   
  data = np.empty((len(results), len(results[0]), max_length))
  for i in range(len(hyps)*5):
      for j in range(len(results[i])):
          data[i, j, :] = results[i][j]

  #calculate the mean and variance along the first axis (over the 5*len(hyps) runs)
  means = np.mean(data, axis=0)
  stds = np.std(data, axis=0)

  #plots
  if task == 'classification':
      fig, axarr = plt.subplots(1, 2)
      plt.subplots_adjust(hspace=0.5)
      fig.suptitle("Learning Curves", fontsize=18, y=0.95)
      fig.set_size_inches(20, 9)
      measure_list=[(means[2], means[3], "Accuracy"), (means[0], means[1], "MSE")]
      stds_list = [(stds[2], stds[3]), (stds[0], stds[1])]
      for measures, ax, dev in zip(measure_list, axarr.ravel(), stds_list):
          ax.plot(measures[0], label="Training Set")
          ax.fill_between(range(len(measures[0])), measures[0]-dev[0], measures[0]+dev[0], alpha=0.2)
          ax.plot(measures[1], label="Test Set", ls='--')
          ax.fill_between(range(len(measures[0])), measures[1]-dev[1], measures[1]+dev[1], hatch='///', facecolor='none', edgecolor='orange', alpha=0.7)
          ax.set_xlabel("epochs", fontsize=14)
          ax.set_ylabel(measures[2], fontsize=14)
          ax.legend(fontsize=16)
      plt.show()
  elif task == 'regression':
      fig, ax = plt.subplots()
      fig.set_size_inches(10, 10)
      ax.plot(means[0], label="Training Set")
      ax.fill_between(range(len(means[0])), means[0]-stds[0], means[0]+stds[0], edgecolor='blue', alpha=0.2)
      ax.plot(means[1], label="Test Set", ls='--')
      ax.fill_between(range(len(means[1])), means[1]-stds[1], means[1]+stds[1], hatch='///', facecolor='none', edgecolor='orange', alpha=0.7)
      ax.set_xlabel("epochs", fontsize=14)
      ax.set_ylabel('MEE', fontsize=14)
      ax.legend(fontsize=16)
      plt.show()

  #computing accuracy/MEE of ensembled model on test set
  labels_ts = [test_set[i][1] for i in range(len(test_set))]
  if task == 'classification':
      res = [abs(out_ts[i]-labels_ts[i]) for i in range(len(out_ts))]
      print("Accuracy on test set: ", round(1 - float(sum(res)/len(out_ts)), 4))
  elif task == 'regression':
      mee = 0
      for i in range(len(labels_ts)):
          mee += np.linalg.norm(out_ts[i]-labels_ts[i])
      print("MEE test set = ", round(mee/len(test_set),4))

  #computing accuracy/MEE of ensembled model on training set
  labels_tr = [training_set[i][1] for i in range(len(training_set))]
  if task == 'classification':
        res = [abs(out_tr[i]-labels_tr[i]) for i in range(len(out_tr))]
        print("Accuracy on training set: ", round(1 - float(sum(res)/len(out_tr)), 4))
  elif task == 'regression':
      mee = 0
      for i in range(len(labels_tr)):
          mee += np.linalg.norm(out_tr[i]-labels_tr[i])
      print("MEE training set = ", round(mee/len(training_set),4))