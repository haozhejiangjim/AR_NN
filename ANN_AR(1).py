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
import numpy as np
import statsmodels.tsa.stattools as smt
import pandas as pd

batch_size = 64
test_batch_size = 128
n_label = 1

# Data Reload
train_dataloader = torch.load('train_dataloader_Beta25.pth')
val_dataloader = torch.load('val_dataloader_Beta25.pth')
test_dataloader = torch.load('test_dataloader_Beta25.pth')
print('*****************************************************Data Loading is done*****************************************************************')
print('\n'*1)

# Create a neutal network
class NN_v4_n9_4(L.LightningModule):

    def __init__(self):
        super().__init__()
        seed_everything(seed=33333)

        self.all_layers = nn.Sequential(
            # 1st hidden layer
            nn.Linear(4,9),
            nn.Tanh(),

            # 2nd hidden layer
            nn.Linear(9, 9),
            nn.Tanh(),

            # 3rd hidden layer
            nn.Linear(9, 9),
            nn.Tanh(),

            # 4th hidden layer
            nn.Linear(9, 9),
            nn.Tanh(),

            # output layer
            nn.Linear(9, 1)
        )
        
    
    def forward(self, input): 
        input = torch.asinh(input)           # IHS transformation
        output = self.all_layers(input)
        return output
        
        
    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr = 0.001)       

    
    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i)
        loss = torch.sum((output_i - label_i)**2, dim=1).mean() / n_label  # here we sum up squared errors and then compute the mean

        ###################
        ##
        ## Logging the loss and the predicted values so we can evaluate the training
        ##
        ###################
        self.log("train_loss", loss)
  
            
        return loss
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_outputs = []
        return 
    
    def validation_step(self, batch, batch_idx):
        input_i, label_i = batch 
        output_i = self.forward(input_i)
        val_result = torch.sum((output_i - label_i)**2, dim=1).mean() / n_label 
        self.val_outputs.append(val_result)
        return val_result
    
    def on_validation_epoch_end(self):
        val_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", val_loss, prog_bar=True)    # here we add our validation loss to monitor in the progress bar
        return {'val_loss': val_loss}                # val_loss is the key which our early stop callback function will monitor
    
    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()           # initialize a test epoch and a list to store the test loss at each batch
        self.test_step_outputs = []
        return
    
    def test_step(self, batch, batch_idx):              # define a test step
        input_i, label_i = batch 
        output_i = self.forward(input_i) 
        loss = torch.sum((output_i - label_i)**2, dim=1).mean() / n_label 
        self.test_step_outputs.append(loss)             # store the test loss for each batch
        return {'test_loss': loss}
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()          # compute the average test loss at the end of this test epoch
        self.log("avg_test_loss", avg_loss)
        return {'avg_test_loss': avg_loss}
    

# train NN 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

model_v4_n9_4 = NN_v4_n9_4()                              

early_stop_callback = EarlyStopping(monitor='val_loss', patience=50, mode='min')       # define the early stopping rule where 'min' means the training stops when the quantity
## stops decreasing. Patience means no. of checks with no improvement after which training will be stopped.

trainer = L.Trainer(max_epochs=200, accelerator="auto", devices="auto",log_every_n_steps=1,  callbacks=[early_stop_callback], check_val_every_n_epoch=1)

trainer.fit(model_v4_n9_4, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(dataloaders=test_dataloader)

print("After optimization, the parameters are...")
for name, param in model_v4_n9_4.named_parameters():
    print(name, param.data)

torch.save(model_v4_n9_4.state_dict(), 'model_Beta25_n9_4.pt')

print('*****************************************************NN Training is done*****************************************************************')
print('\n'*1)