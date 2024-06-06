import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import Adam 

import lightning as L # lightning has tons of cool tools that make neural networks easier
from torch.utils.data import TensorDataset, DataLoader

from lightning_lite.utilities.seed import seed_everything
import seaborn as sns

# load data 
from statsmodels.tsa.arima_process import ArmaProcess
import random
import numpy as np
import statsmodels.tsa.stattools as smt
import pandas as pd

batch_size = 64
test_batch_size = 128
n_label = 1

# Initialize empty lists to store input sequences and labels
X = []  # Input sequences
y = []  # Corresponding labels
temp = []
num_simulation = 50000
seed_everything(seed=33)
random_phi = np.random.beta(0.25, 0.25, num_simulation)            # beta-distributed autoregressive parameter
random_size = np.random.exponential(5, num_simulation)             # exponentially distributed sample size
# Create input sequences and labels
for i in range(num_simulation):    # no. of simulations
    # Create a random phi for each following series
    phi = random_phi[i] * 2 - 1  # this is also called rho  
    intercept = 0 
    n_seq = int(np.round((random_size[i] + 0.1) * 100)) # generate a random sequence size
    
    # Draw a random AR(1) series following the phi as above
    ar1 = np.array([1, -phi])    # use -phi because this function uses the lag-polynomial representation
    ma1 = np.array([1])
    AR_object = ArmaProcess(ar1, ma1)
    series = AR_object.generate_sample(nsample=n_seq, scale=1, burnin = 500) + intercept  

    mean = np.mean(series[:n_seq])
    var = np.var(series[:n_seq], ddof=1)
    autocov = smt.acovf(series[:n_seq], adjusted=True)[1]   

    input = [mean, autocov, var, int(n_seq**.5)]
    label = [phi] 

    X.append(input)
    y.append(label)

    data = list(zip(X, y))                 # merge the input and label


# Create training, validation and testing sets from the data
split_ratio, val_ratio = 0.8, 0.9                          # 80% for training, 10% for validation and 10% for testing
split_index, val_index = int(len(data) * split_ratio), int(len(data) * val_ratio)
train_data, train_label = zip(*data[:split_index])
val_data, val_label = zip(*data[split_index: val_index])
test_data, test_label = zip(*data[val_index:])

inputs_train = torch.tensor(train_data, dtype=torch.float32)
inputs_val = torch.tensor(val_data, dtype=torch.float32)
inputs_test = torch.tensor(test_data, dtype=torch.float32)

labels_train = torch.tensor(train_label, dtype=torch.float32)
labels_val = torch.tensor(val_label, dtype=torch.float32)
labels_test = torch.tensor(test_label, dtype=torch.float32)

train_dataset = TensorDataset(inputs_train, labels_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(inputs_val, labels_val)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(inputs_test, labels_test)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

torch.save(train_dataloader, 'train_dataloader_Beta25.pth')
torch.save(val_dataloader, 'val_dataloader_Beta25.pth')
torch.save(test_dataloader, 'test_dataloader_Beta25.pth')
print('*****************************************************Data Loading is done*****************************************************************')
print('\n'*3)


