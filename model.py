import torch.nn as nn
import torch

class OneLayerModel(nn.Module):
    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.fc = nn.Linear(1*32*64, 100)

    def forward(self, x, activation_func):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.log_softmax(x, dim=1)
        return x


class TwoLayerModel(nn.Module):
    def __init__(self, neurons):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(1*32*64, neurons)
        self.fc2 = nn.Linear(neurons, 100)

    def forward(self, x, func):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = func(x)
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)
        return x


class ThreeLayerModel(nn.Module):
    def __init__(self, neurons):
        super(ThreeLayerModel, self).__init__()
        self.fc1 = nn.Linear(1*32*64, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 100)

    def forward(self, x, func):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = func(x)
        x = self.fc2(x)
        x = func(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x
