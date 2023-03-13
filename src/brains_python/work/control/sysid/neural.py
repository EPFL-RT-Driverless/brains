import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset
from torch.autograd import Variable


class ODE(nn.Module):
    """
    This residual multilayer perceptron approximates the following parameters: F_x, F_y_R, F_y_F, I_z, v_x1, v_x2
    from the following input: x=(X, Y, phi, v_x, v_y, r, T, delta), u=(dT, ddelta)
    """

    def __init__(self, n_input=10, n_output=6, n_hidden=(32, 32)) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(n_input, n_hidden[0]))
        for i in range(len(n_hidden) - 1):
            self.hidden_layers.append(nn.Linear(n_hidden[i], n_hidden[i + 1]))
        self.output_layer = nn.Linear(n_hidden[-1], n_output)

    def forward(self, x, u):
        z = torch.cat((x, u), dim=1)
        z = self.relu(self.hidden_layers[0](z)) + z
        for i in range(1, len(self.hidden_layers)):
            z = self.relu(self.hidden_layers[i](z)) + z
        z = self.output_layer(z)
        return z


class RK4(nn.Module):
    """
    This encapsulates another neural network and implements the Runge-Kutta 4th order method.
    """

    def __init__(self, ode=ODE()) -> None:
        super().__init__()
        self.ode = ode

    def forward(self, x, u, dt):
        k1 = self.ode(torch.cat((x, u), dim=1))
        k2 = self.ode(torch.cat((x + 0.5 * dt * k1, u), dim=1))
        k3 = self.ode(torch.cat((x + 0.5 * dt * k2, u), dim=1))
        k4 = self.ode(torch.cat((x + dt * k3, u), dim=1))
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
