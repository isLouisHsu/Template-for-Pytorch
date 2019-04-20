"""
define your main function here
"""
from config import configer
from trainer import Trainer
from datasets import SineData

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn

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
    trainer = Trainer(configer, net, SineData(), SineData(), nn.MSELoss(), SGD, MultiStepLR, num_to_keep=5, resume=False)
    trainer.train()

if __name__ == "__main__":
    main()