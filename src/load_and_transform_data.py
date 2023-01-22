import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def encode_dataset(dataset,
                   input_columns = [1,6],
                   target_columns = [0,1], 
                   categorial_input = True):
    """Function that given a dataframe with input and output columns
    return a dataset list where each element is in shape of <input, output>
    with optional one-hot encoding of the input"""
    start_input_column = min(input_columns)
    total_columns = max(input_columns)
    start_output_column = min(target_columns)
    total_output_columns = max(target_columns)
    ds = [] #initialize the dataset, its element will be in shape of <input, output>
    if categorial_input:
        dd = {} #initialize a dictionary cointaning for every column (input component) 
                #the number of distinct elements for one-hot encoding of each column
        i = 0
        for column in range(total_columns):
            m = max(set(dataset.iloc[:, start_input_column+column]))
            dd[i] = m #assign the number of distinct elements for each column
            i += 1
        input_range = np.sum([v for v in dd.values()]) #length of the future one-hot vector for all the columns
        
    for line in range(len(dataset)):
        if categorial_input:
            input_ = np.zeros(input_range) #initialize a 0 vector
            l = list(dataset.iloc[line, start_input_column:total_columns+start_input_column]) #select the input columns from the dataframe
            i = 0
            index = 0 #index of the one-hot slots encoding each column
            for value in l:
                input_[index+value-1] = 1 #assign 1 to the binary slot corrisponding to the current value of the column
                index += dd[i] #update the index variable to the next column
                i += 1 #update the index of the column for future access on the dd dictionary to select the number of slots for each column
        else:
            input_= np.array(dataset.iloc[line, start_input_column:total_columns + start_input_column], dtype='object') #select the input columns from the dataframe
        target = np.array(dataset.iloc[line, start_output_column:start_output_column + total_output_columns], dtype='object') #select the output columns from the dataframe
        example=input_,target
        ds.append(example)
    return ds

def load_dataset (path, i=1, MLCUPTR=True):
  """Loads training and test set of Monk_i if i is given, 
  otherwise loads ML-CUP22 dataset either training set or 
  test set given the value of the variable <MLCUPTR>"""
  if i in [1, 2, 3] :
      train = pd.read_csv(path+"/monks"+str(i)+"_training.csv")
      training_set = encode_dataset(train)
      test = pd.read_csv(path+"/monks"+str(i)+"_test.csv")
      test_set = encode_dataset(test)
  else:
    scaler = MinMaxScaler() #initialize the scaler
    if MLCUPTR:
      df = pd.read_csv(path+"/ML-CUP22-TR.csv", comment='#', skiprows=7, header=None)
      df = df.drop(columns = 0)
      #returns the training set and an internal test set
      #with respectively 70% and 30% splitting
      train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(train_df.iloc[:,0:9].values) #apply data normalization for the train set
      X_test = scaler.fit_transform(test_df.iloc[:,0:9].values) #apply data normalization for the internal test set
      training_set = []
      test_set = []
      #encode the training set in shape of a list of <input, output> elements
      for i in range(len(X_train)):
          training_set += [[np.array(X_train[:, 0:9][i]),np.array((train_df.iloc[:,-2:].values)[i])]]
      #encode the internal test set in shape of a list of <input, output> elements
      for i in range(len(X_test)):
          test_set += [[np.array(X_test[:, 0:9][i]),np.array((test_df.iloc[:,-2:].values)[i])]]
    else:
      df = pd.read_csv(path+"/ML-CUP22-TS.csv", comment='#', skiprows=7, header=None)
      df = df.drop(columns = 0)
      X_test = scaler.fit_transform(df.iloc[:,0:9].values) #apply data normalization for the test set
      #encode the test set in shape of a list of <input, output> elements
      for i in range(len(X_test)):
          test_set += [[np.array(X_test[:, 0:9][i]),np.array((test_df.iloc[:,-2:].values)[i])]]
      training_set=[]
  return training_set, test_set