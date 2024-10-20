import random

import numpy as np
import pandas as pd
import scipy

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class trajectory_dataset(Dataset) :
    def __init__(self, opt, csv_path="dataset/HCMC-vehicle-dataset.csv") :
        # Inheritance
        super(trajectory_dataset, self).__init__()

        # Initialize Variable
        self.opt = opt
        self.csv_path = csv_path

        # Function-Calling
        self.load_data()
        self.input, self.target, self.max = self.preprocess_data()

    def __len__(self) :
        return len(self.input)

    def __getitem__(self, index) :
        # Reshape Input & Target Data
        input = self.input.iloc[index,:].values.reshape(-1, 1)
        target = self.target.iloc[index,:].values.reshape(1, -1)

        # Create Numpy Array Instance
        input_bbox, target_bbox = np.zeros((input.shape[0]//2, 1)), np.zeros((1, target.shape[1]//2))
        
        # Save Center Coordinate
        for i in range(input.shape[0]//2) :
            input_bbox[i,0] = (input[i*2,:] + input[i*2+1,:])/2
        
        for i in range(target.shape[1]//2) :
            target_bbox[0,i] = (target[:,i*2] + target[:,i*2+1])/2
        
        # Convert Numpy Array to PyTorch Tensor
        input_bbox = torch.tensor(input_bbox, dtype=torch.float32)
        target_bbox = torch.tensor(target_bbox, dtype=torch.float32)

        return (input_bbox, target_bbox)

    def load_data(self) :
        self.data = pd.read_csv(self.csv_path).iloc[:,4:]

    def preprocess_data(self) :
        # Get Mininum & Maximum Data
        max = np.max(self.data)
        
        # Apply Min-Max Norm
        self.data = self.data/max
        
        # Split Input & Target Data
        target_column = ["X_max_8", "X_min_8", "Y_max_8", "Y_min_8"]
        input = self.data.drop(target_column, axis=1)
        target = self.data[target_column]

        return input, target, max


def get_dataloader(opt, csv_path="dataset/HCMC-vehicle-dataset.csv") :
    # Load Dataset
    dataset = trajectory_dataset(opt, csv_path)

    # Split Dataset
    num_train = int(dataset.__len__()*opt.train_ratio)
    num_test = int(dataset.__len__()*opt.test_ratio) - num_train
    num_valid = int(dataset.__len__() - num_test - num_train)
    train, valid, test = random_split(dataset, [num_train, num_valid, num_test])

    # Create Dataloader Instance
    train_loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, dataset