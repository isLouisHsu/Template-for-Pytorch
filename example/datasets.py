import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class SineData(Dataset):

    def __init__(self):

        f = lambda x: np.sin(x) + np.random.randn(x.shape[0])*0.2
        x = np.linspace(-5, 5, 1000)
        y = f(x)
        self.samples = np.concatenate([x, y]).reshape((2, -1)).T

    def __getitem__(self, index):

        X_y = self.samples[index].reshape((-1, 1))
        X = X_y[0]; y = X_y[1]
        return X, y

    def __len__(self):

        return self.samples.shape[0] 
     