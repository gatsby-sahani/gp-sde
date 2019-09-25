import torch
from torch import nn


class Exp(nn.Module):
    """
    Exponential non-linearity
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.name = 'exponential'

    def derivative(self, x):
        return torch.exp(x)

    def forward(self, x):
        return torch.exp(x)


class Relu(nn.Module):
    """
    Rectified linear non-linearity
    """

    def __init__(self, a):
        super(Relu, self).__init__()
        self.floor = a
        self.name = 'relu'

    def derivative(self, x):
        return torch.double(torch.x > self.floor)

    def forward(self, x):
        return torch.max(x, self.floor)


class Sigmoid(nn.Module):
    """
    Sigmoidal non-linearity
    """

    def __init__(self, a, b):
        super(Sigmoid, self).__init__()
        self.slope = a
        self.max = b
        self.name = 'sigmoid'

    def derivative(self, x):
        return self.max * self.slope / (1 + torch.exp(-self.slope * x))**2

    def forward(self, x):
        return self.max / (1 + torch.exp(-self.slope * x))
