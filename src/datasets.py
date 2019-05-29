import os
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Data(Dataset):

    def __init__(self):

        pass

    def __getitem__(self, index):

        pass

    def __len__(self):

        return 

class Prefetcher():

    def __init__(self, dataloader):

        self.dataloader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self._X, self._y = None, None
        self._preload()
    
    def _preload(self):

        try:
            self._X, self._y = next(self.dataloader)
        except StopIteration:
            self._X, self._y = None, None
            return
        
        with torch.cuda.stream(self.stream):
            self._X = self._X.cuda(non_blocking=True)
            self._y = self._y.cuda(non_blocking=True)
    
    def next(self):
        
        torch.cuda.current_stream().wait_stream(self.stream)
        X = self._X; y = self._y
        self._preload()
        
        return X, y