import torch
from torchvision import transforms, datasets

import numpy as np

from config import *

def train_test_split(data_dir, valid_size=0.2, batch_size = batch_size):
    train_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0))
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor()#,
        #transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0))
    ])

    train_data = datasets.ImageFolder(data_dir, train_transforms)
    test_data = datasets.ImageFolder(data_dir, test_transforms)

    num_train = len(train_data)
    indicies = list(range(num_train))
    split = int(np.floor(valid_size*num_train))
    np.random.shuffle(indicies)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indicies[split:], indicies[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_loader, test_loader