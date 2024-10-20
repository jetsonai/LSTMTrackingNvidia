from ast import literal_eval

import pandas as pd

import torch
from torch.utils.data import Dataset


class LoadDataset(Dataset) :
    def __init__(self, opt, for_val) :
        # Inheritance
        super(LoadDataset, self).__init__()

        # Initialize Variable
        self.opt = opt
        
        # Load CSV
        self.data = self.load_csv("dataset/csv/val_dataset.csv" if for_val else "dataset/csv/train_dataset.csv")
        
    def load_csv(self, csv_dir) :
        # Convert into Pillow Image
        data = pd.read_csv(csv_dir)

        return data

    def __getitem__(self, index) :
        # Split Sequence
        input, target = self.data.iloc[index][:self.opt.input_frame], self.data.iloc[index][self.opt.input_frame:]
        
        # Convert to PyTorch Tensor
        input_x, input_y = [literal_eval(i)[0] for i in input], [literal_eval(i)[1] for i in input]
        input_x, input_y = torch.tensor(input_x).unsqueeze(-1), torch.tensor(input_y).unsqueeze(-1)

        target_x, target_y = [literal_eval(i)[0] for i in target], [literal_eval(i)[1] for i in target]
        target_x, target_y = torch.tensor(target_x).unsqueeze(-1), torch.tensor(target_y).unsqueeze(-1)
        
        # Apply Min-Max Normalization
        input_x, input_y = self.min_max_norm(input_x, 0, self.opt.video_size), self.min_max_norm(input_y, 0, self.opt.video_size)
        target_x, target_y = self.min_max_norm(target_x, 0, self.opt.video_size), self.min_max_norm(target_y, 0, self.opt.video_size)
        
        # Concatenate Tensor ([#Seq, 2])
        input = torch.cat([input_x, input_y], dim=-1)
        target = torch.cat([target_x, target_y], dim=-1)
        
        return input, target

    def __len__(self) :
        # Get Number of Data
        return self.data.shape[0]
    
    def min_max_norm(self, input, min, max) :
        output = (input-min)/(max-min)
        
        return output