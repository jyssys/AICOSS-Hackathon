import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from config import configs

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.train_len = int(len(self.data_frame) * 0.8)

    def __len__(self):
        if self.mode == 'train':
            return self.train_len
        elif self.mode == 'val':
            return len(self.data_frame) - self.train_len
        else:
            return len(self.data_frame)

    def __getitem__(self, idx):
        if self.mode == 'val':
            idx += self.train_len

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1].split('/')[-1])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mode in ['train', 'val']:
            labels = self.data_frame.iloc[idx, 2:].values.astype('float')
            return {'image': image, 'labels': torch.tensor(labels)}
        else:
            return {'image': image}

def get_transforms(transform_resize, is_crop):
    if is_crop:
        train_transform = transforms.Compose([
            transforms.Resize((transform_resize, transform_resize)),
            transforms.RandomCrop(transform_resize, padding=int((transform_resize/32)*4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((transform_resize, transform_resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((transform_resize, transform_resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform

def get_data_loaders(batch_size, train_transform, test_transform):
    train_dataset = CustomDataset(csv_file='data/train.csv', root_dir='data/train/train', transform=train_transform, mode='train')
    val_dataset = CustomDataset(csv_file='data/train.csv', root_dir='data/train/train', transform=test_transform, mode='val')
    test_dataset = CustomDataset(csv_file='data/test.csv', root_dir='data/test/test', transform=test_transform, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader