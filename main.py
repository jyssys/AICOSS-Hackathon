import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn import Sigmoid
from torch.optim.lr_scheduler import StepLR
# from data_loader import CustomDataset
from model import Resnet18, Resnet50, EfficientNet, EfficientNetB3, EfficientNetB4, EfficientNetB5
from utils import SAM, mixup, cutmix
from method import train_model, val_model, predict
from config import configs
from ensemble import ensemble

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
    
def dir_maker(dirName):
    if not os.path.exists(dirName):
        os.makedirs(dirName)

def save_predictions_to_csv(predictions, file_path, version):
    submission_df = pd.read_csv(file_path)
    submission_df.iloc[:, 1:] = predictions
    submission_df.to_csv(f'results/updated_submission_{version+1}.csv', index=False)

def main(batch_size, transform_resize, is_crop, epoch, num):
    print(f'now version : {num}')
    # define seed
    seed_everything(605)
    
    # define mps or cpu/cuda gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    gc.collect()
    torch.cuda.empty_cache()
    
    # define model
    num_classes = 60  # 실제 클래스 수에 맞게 설정
    model = EfficientNetB3(num_classes)
    model.to(device)

    # loss function & optimizer
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    base_optimizer = optim.Adam
    SAM_optimizer = SAM(model.parameters(), base_optimizer, lr = 0.001, rho = 0.05)

    # scheduler
    scheduler = StepLR(SAM_optimizer, step_size=5, gamma=0.1)
 
    # train
    infer_model = train_model(model, criterion, SAM_optimizer, scheduler, device, num, num_epochs = config['epoch'], mode = 'sam_normal')
    torch.save(infer_model.state_dict(), f'weights/model_config_{i+1}.pth')

    # pred
    pred = predict(infer_model, device, num)

    # save
    save_predictions_to_csv(pred, 'data/sample_submission.csv', i)
    print(f'save complete. watch data/updated_submission.csv')

if __name__ == '__main__':
    dir_maker('weights')
    dir_maker('results')
    
    for i, config in enumerate(configs):
        print(f"Running configuration {i+1}/{len(configs)}: {config}")
        
        main(**config)
        
    ensemble()