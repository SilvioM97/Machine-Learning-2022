# NNfromScratch-py

This repository contains a Python framework for training and testing neural networks from scratch. Cross validation could run in parallel for efficient validation

## Usage

1. **Load the Dataset:**

   - Make sure you have the dataset available. If not, obtain the dataset or create your own.
   - Update the code in the `load_dataset()` function to load your specific dataset.

2. **Hyperparameters:**

   Configure the hyperparameters to validate with cross-validation in the `hyperparameters` dictionary within the code. Here's a brief explanation of each hyperparameter:

   - `architecture`: The architecture of the neural network, specified as a list of integers representing the number of neurons in each layer.
   - `xavier`: Whether to use Xavier initialization for weight initialization.
   - `activation`: The activation function to be used in the neural network (relu or sigmoid).
   - `alpha_mom`: The momentum factor used in gradient descent optimization.
   - `lambda_reg`: The regularization parameter used for weight regularization.
   - `mb`: The mini-batch size used during training.
   - `eta`: The learning rate used in gradient descent optimization.
   - `epochs`: The number of training epochs.
   - `eta_decay`: Whether to apply learning rate decay during training.
  Example:
   ```python
   hyperparameters = {
       "architecture": [[17, 20, 1], [17, 5, 10, 1]],
       'xavier': [False, True],
       'activation': ['relu', 'sigmoid'],
       "alpha_mom": [0.15, 0.1, 0.05],
       "lambda_reg": [1e-5, 1e-6, 1e-7],
       "mb": [25, 50, 100, 200],
       "eta": [0.5, 0.75, 1],
       "epochs": [200, 300, 500],
       "eta_decay": [False, True]
   } 
   ```

3. **Cross Validation:**

   - Adjust the cross-validation settings (number of folds, repetitions) in the `cross_validation()` function call.
   
   Example:

   ```python
   hyps = cross_validation(dataset, hyperparameters, K, cross_times, task="classification", max_workers=40, random_size)
    ```
    
 4. **Testing:**

   Once you have obtained the best hyperparameter settings from the cross-validation step, you can proceed to the testing phase. Follow these steps:

   - Define the hyperparameters' dictionaries you want to Ensemble for the test.

Example:
   
   ```python
   hyps = [{'architecture': [17, 3, 1], 'xavier': True, 'activation': 'sigmoid', 'alpha_mom': 0.1, 'lambda_reg': 1e-07, 'mb': 25, 'eta': 0.5, 'epochs': 600, 'eta_decay': False},
           {'architecture': [17, 3, 1], 'xavier': False, 'activation': 'relu', 'alpha_mom': 0.05, 'lambda_reg': 1e-06, 'mb': 100, 'eta': 0.5, 'epochs': 600, 'eta_decay': False},
           {'architecture': [17, 3, 1], 'xavier': False, 'activation': 'relu', 'alpha_mom': 0.1, 'lambda_reg': 1e-06, 'mb': 100, 'eta': 0.75, 'epochs': 600, 'eta_decay': False},
           {'architecture': [17, 3, 1], 'xavier': True, 'activation': 'sigmoid', 'alpha_mom': 0.15, 'lambda_reg': 1e-06, 'mb': 25, 'eta': 0.75, 'epochs': 600, 'eta_decay': False},
           {'architecture': [17, 3, 1], 'xavier': False, 'activation': 'sigmoid', 'alpha_mom': 0.15, 'lambda_reg': 1e-06, 'mb': 50, 'eta': 0.75, 'epochs': 600, 'eta_decay': False}]
  ```
    
  - Run the tests:
   ```python
  tests(training_set, test_set, hyps, 'classification')
  ```

   
