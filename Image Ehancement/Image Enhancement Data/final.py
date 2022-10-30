import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
from torch import nn
from matplotlib import pyplot as plt

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

#from models import *
#from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class convBlock(nn.Module):
    def __init__(self):
        super(convBlock, self).__init__()
        self.conv_in = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv_tanh = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
 

    
    def forward(self, x):
        x1 = self.conv_in(x)

        x2 = self.conv_relu(x1) + self.conv_tanh(x1)

        x3 = torch.add(x1, x2) 
        x3 = nn.functional.relu(x3)

        x4 = self.conv_relu(x3) + self.conv_tanh(x3)

        x5 = torch.add(x3, x4)
        out = nn.functional.relu(x5)

        return out


    
    
class out(nn.Module):
    def __init__(self, in_channels, CONV_BLOCKS):
        super(out, self).__init__()
        
        #conv block for input
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            )
        #Residual Block
        conv_blocks = []
        for _ in range(CONV_BLOCKS):
            conv_blocks.append(convBlock())
        self.conv_blocks = nn.Sequential(*conv_blocks)

        #conv block for output
        self.conv_out = nn.Sequential(nn.Conv2d(32, in_channels, kernel_size=1, stride=1))
    
    def forward(self, x):
        x = self.conv_in(x)

        '''for i in range(self.CONV_BLOCKS):
          if self.RESIDUAL:
            x = convBlock(x)
          else:
            x = convBlockwoResidual(x)'''
        x1 = self.conv_blocks(x)

        out = self.conv_out(x1)
        return out