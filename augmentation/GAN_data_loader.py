import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

import pandas as pd
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, indices=None, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        if indices is not None:
            self.data_frame = self.data_frame.iloc[indices]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1].split('/')[-1])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        labels = self.data_frame.iloc[idx, 2:].values.astype('float')
        return {'image': image, 'labels': torch.tensor(labels)}

# data preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train, val & test set load
dataset = CustomDataset(csv_file='data/train.csv', root_dir='data/train', transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)


# ----------------------------------------------------------------------------------------
# data = pd.read_csv('data/train.csv')

# X = data.index.values
# y = data.iloc[:,57]

# train_idx, val_idx = train_test_split(X, test_size=0.2, stratify=y, random_state=605)


# train_dataset = CustomDataset(csv_file='data/train.csv', root_dir='data/train', indices=train_idx, transform=train_transform)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# val_dataset = CustomDataset(csv_file='data/train.csv', root_dir='data/train', indices=val_idx, transform=test_transform)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# test_dataset = CustomDataset(csv_file='data/test.csv', root_dir='data/test', transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)