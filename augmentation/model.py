import torch
import torch.nn as nn
import torch.optim as optim

from GAN import *
from GAN_method import train_model

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')
import time
import gc
import logging
import random
import os

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    dir_name = "data/CGAN_train"
    # define seed
    seed_everything(605)
    
    # define mps or cpu/cuda gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    gc.collect()
    torch.cuda.empty_cache()

    # define model
    num_classes = 60
    
    # Initialize generator/Discriminator
    discriminator = Discriminator()
    generator = Generator()

    # Device setting
    discriminator = discriminator.to(device)
    generator = generator.to(device)

    # Loss function & Optimizer setting
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    
    train_model(discriminator, generator, criterion, g_optimizer, d_optimizer, device, 20, 128, dir_name)

if __name__ == '__main__':
    main()