import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from config.config import Config
from dataloader.dataloder import DataLoader



class Trainer:
    def __init__(self, config_path):
        self.config = Config(config_path=config_path)
        self.dataloader = DataLoader()