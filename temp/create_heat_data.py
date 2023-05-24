## Imports 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer


class CustomDataset(Dataset):
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, index):
        return self.t[index], self.x[index]
    

def create_heat_data(nx=100, nt=100, alpha=0.05, n=1):

    h = 1 / nx ## Step size in x direction
    k = 1 / nt ## Step size in t direction
    r =  alpha * k / h**2
    X, T = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, nt + 1))

    ## Output
    data_in = []
    data_out = []

    A = np.zeros((nx - 1, nx - 1))
    B = np.zeros((nx - 1, nx - 1))

    for i in range(nx - 1):
        A[i, i] = 2 + 2 * r
        B[i, i] = 2 - 2 * r

    for i in range(nx - 2):
        A[i + 1, i] = -r
        A[i, i + 1] = -r
        B[i + 1, i] = r
        B[i, i + 1] = r

    Ainv = np.linalg.inv(A)

    ## Generate Data
    for _ in range(n):
        
        beta = np.random.uniform(-10, 10)
        beta_vec = beta * np.ones(101)

        u0 = np.sin(beta * X[0])

        data_in.append([beta_vec , np.arange(0, 1.01, 0.01)])

        ## Create Solution Matrix
        u = np.zeros((nt + 1, nx + 1))
        u[0] = u0
        u[:, 0] = 0
        u[:, -1] = 0

        # b = np.zeros(nx - 1)
        for j in range(1, nt + 1):
            # b[0] = r * u[j - 1, 0] + r * u[j, 0]
            # b[-1] = r * u[j - 1, -1] + r * u[j, -1]
            u[j, 1:nx] = Ainv @ ((B @ u[j - 1, 1:nx])) # + b)

        data_out.append(u[-1])

    data_in = np.array(torch.tensor(data_in).float())
    data_out = np.array(torch.tensor(data_out).float().unsqueeze(1))

    return CustomDataset(data_in, data_out)