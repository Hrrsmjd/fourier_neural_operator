"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""


import torch.nn.functional as F
from timeit import default_timer
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d Fourier Integral Operator
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)) ## Tensor

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        # print(f'Before FFT: {x.shape}')
        x_ft = torch.fft.rfft(x)
        # print(f'After FFT: {x_ft.shape}')

        # print(f'Weigths shape: {self.weights.shape}')

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # print(f'x_ft[0, :, :self.modes]: {x_ft[0:1, :, :self.modes].shape}')
        # print(f'Complex mul1d shape: {self.compl_mul1d(x_ft[0:1, :, :self.modes], self.weights).shape}')
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        # print(f'After linear trasnformation: {out_ft.shape}')
        # print(f'{out_ft[0, 0, :]}')

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        # print(f'After iFFT: {x.shape}')
        return x


class MLP(nn.Module): ##TODO: Rewrite
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


################################################################
#  1d Fourier Network
################################################################
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(1, self.width) # input channel_dim is 2: (u0(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # print(grid.shape)
        # print(f'x_shape:{x.shape}')
        # x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 2, 1) ##
        # print(f'Before P x_shape:{x.shape}')
        x = self.p(x)
        # print(f'After P x_shape:{x.shape}')
        x = x.permute(0, 2, 1)
        # print(f'After permute x_shape:{x.shape}')
        
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        # print(f'x1_shape:{x1.shape}')
        x1 = self.mlp0(x1)
        # print(f'x1_shape:{x1.shape}')
        x2 = self.w0(x)
        # print(f'x2_shape:{x2.shape}')
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


################################################################
#  configurations
################################################################
# ntrain = 1000
# ntest = 100

# sub = 2**3 #subsampling rate
# h = 2**13 // sub #total grid size divided by the subsampling rate
# s = h

batch_size = 64
learning_rate = 0.001
# epochs = 500
epochs = 20
# iterations = epochs*(ntrain//batch_size)

modes = 1
width = 64

################################################################
# read data
################################################################

# # Data is of the shape (number of samples, grid size)
# dataloader = MatReader('data/burgers_data_R10.mat')
# x_data = dataloader.read_field('a')[:,::sub]
# print(x_data.shape)
# print(x_data.shape[0])
# y_data = dataloader.read_field('u')[:,::sub]
# print(y_data.shape)
# print(y_data.shape[0])

# x_train = x_data[:ntrain,:]
# y_train = y_data[:ntrain,:]
# x_test = x_data[-ntest:,:]
# y_test = y_data[-ntest:,:]

# x_train = x_train.reshape(ntrain,s,1)
# x_test = x_test.reshape(ntest,s,1)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

## Simple Harmonic Oscillator
def simple_harmonic_oscillator(k: int, m: int, x0: int, v0: int, t: np.array) -> np.array:
    x = x0 * np.cos(np.sqrt(k / m) * t) + v0 / np.sqrt(k / m) * np.sin(np.sqrt(k / m) * t)
    return x    

## Number of Samples 
n = 100000
## Output
data_t = []
data_x = []

# k = np.random.randint(1, 100)
# m = np.random.randint(1, 100)
# x0 = np.random.normal(0, 10)
# v0 = np.random.normal(0, 10)
k = 1
m = 1000
x0 = 0.1
v0 = 0.1

## Generate Data
for _ in range(n):
    # k = np.random.randint(1, 100)
    # m = np.random.randint(1, 100)
    # x0 = np.random.normal(0, 10)
    # v0 = np.random.normal(0, 10)
    
    t = np.random.uniform(0, 100)
    data_t.append(t)
    data_x.append(simple_harmonic_oscillator(k, m, x0, v0, t))

data_t = np.array(torch.tensor(data_t).float().unsqueeze(1).unsqueeze(1))
data_x = np.array(torch.tensor(data_x).float().unsqueeze(1).unsqueeze(1))

## Data Loader
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, t, x):
        self.t = t
        self.x = x

    def __len__(self):
        return len(self.t)
    
    def __getitem__(self, index):
        return self.t[index], self.x[index]

data = CustomDataset(data_t, data_x)
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# model
# model = FNO1d(modes, width).cuda()
model = FNO1d(modes, width).cpu()
print(count_params(model))

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

# myloss = LpLoss(size_average=False)
myloss = nn.MSELoss()

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
    train_loop(train_loader, model, myloss, optimizer)
print("Done!")

t_test= torch.from_numpy(np.linspace(0, 100, 100)).float().unsqueeze(0).unsqueeze(0)
model(t_test.permute(2, 1, 0)).detach()

# for ep in range(epochs):
#     model.train()
#     t1 = default_timer()
#     train_mse = 0
#     train_l2 = 0
#     for x, y in train_loader:
#         # x, y = x.cuda(), y.cuda()
#         x, y = x.cpu(), y.cpu()

#         optimizer.zero_grad()
#         out = model(x)

#         mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
#         l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
#         l2.backward() # use the l2 relative loss

#         optimizer.step()
#         scheduler.step()
#         train_mse += mse.item()
#         train_l2 += l2.item()

#     model.eval()
#     test_l2 = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             # x, y = x.cuda(), y.cuda()
#             x, y = x.cpu(), y.cpu()

#             out = model(x)
#             test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

#     train_mse /= len(train_loader)
#     train_l2 /= ntrain
#     test_l2 /= ntest

#     t2 = default_timer()
#     print(ep, t2-t1, train_mse, train_l2, test_l2)

# # torch.save(model, 'model/ns_fourier_burgers')
# pred = torch.zeros(y_test.shape)
# index = 0
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# with torch.no_grad():
#     for x, y in test_loader:
#         test_l2 = 0
#         # x, y = x.cuda(), y.cuda()
#         x, y = x.cpu(), y.cpu()

#         out = model(x).view(-1)
#         pred[index] = out

#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         print(index, test_l2)
#         index = index + 1

# scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})
# np.savetxt('pred/burger_test.csv', pred.cpu().numpy(), delimiter=",")
