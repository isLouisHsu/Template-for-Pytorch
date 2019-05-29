"""
define your main function here
"""
from config import configer
from trainer import Trainer
from datasets import SineData

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

from matplotlib import pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.f = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):

        x = self.f(x)
        return x


def main():
    net = Net()
    trainset = SineData()
    validset = SineData()

    x = validset.samples[:, 0]
    y_pred_init = net(torch.tensor(x.reshape((-1, 1))).float()).detach().numpy().reshape(-1)

    trainer = Trainer(configer, net, net.parameters(), trainset, validset, nn.MSELoss(), SGD, MultiStepLR, num_to_keep=5, resume=False)
    trainer.train()

    y_true = validset.samples[:, 1]
    y_pred_trained = net(torch.tensor(x.reshape((-1, 1))).float()).detach().numpy().reshape(-1)

    plt.figure()
    plt.plot(x, y_true, c='r')
    plt.plot(x, y_pred_init, c='g')
    plt.plot(x, y_pred_trained, c='b')
    plt.show()


if __name__ == "__main__":
    main()