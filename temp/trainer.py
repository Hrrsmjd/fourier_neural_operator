## Imports 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer

from create_heat_data import create_heat_data
from FNO1D import FourierNetworkOperator1D
from utils import count_parameters


## Create dataloader
data = create_heat_data(n=100)
dataloader = DataLoader(data, batch_size=16, shuffle=True)

## Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')

model = FourierNetworkOperator1D(2, 1, width=64, modes=4)
count_parameters(model)

## Training Parameters
learning_rate = 1e-3
epochs = 5

## Loss Function
loss_function = nn.MSELoss()

## Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Training Loop
def train_loop(dataloader, model, loss_function, optimizer):
    size = len(dataloader.dataset)
    for batch, (t, x) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(t)
        loss = loss_function(pred, x)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(t)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss_function, optimizer)
print("Done!")

torch.save(model.state_dict(), "tempmodel.pth")