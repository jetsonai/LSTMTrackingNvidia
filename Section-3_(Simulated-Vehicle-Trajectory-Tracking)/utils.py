import random

import numpy as np

import torch


def fix_seed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class average_meter(object) :
    def __init__(self) :
        self.reset()

    def reset(self) :
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) :
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
