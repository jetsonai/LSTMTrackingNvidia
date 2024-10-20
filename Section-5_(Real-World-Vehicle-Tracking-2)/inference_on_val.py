from os import makedirs

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import LoadDataset
from model import LSTM
from utils import fix_seed
from config import read_all_arguments


def main(opt) :
    # Fix Seed
    fix_seed(opt.seed)
    
    # Load Validation Dataset
    val_dataset = LoadDataset(opt, for_val=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=False, shuffle=False)
    
    # Fix Seed
    fix_seed(opt.seed)
    
    # Create Model Instance
    model = LSTM(opt)
    
    # Load Pretraind Weight
    model.load_state_dict(torch.load(f"ckpt/lstm/best.pth"), strict=True)
    
    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type : {device}")
    
    # Assign Device
    model = model.to(device)
    
    # Create Directory
    graph_dir = "result/val"
    makedirs(graph_dir, exist_ok=True)
    
    # Start Test Phase
    val_bar = tqdm(val_loader)

    # Start Test Phase
    for index, data in enumerate(val_bar) :
        # Load Dataset
        input, target = data
        
        # Assign Device
        input, target = input.to(device), target.to(device)
        
        with torch.no_grad() :
            # Get Prediction
            pred = model(input)
            
            # Affine Transformation
            input = (input.detach().cpu().numpy()*639).astype("int32")
            
            # Affine Transformation
            pred = (pred[:,-opt.target_frame:,:].clamp(0,1).detach().cpu().numpy()*639).astype("int32")
    
            # Affine Transformation
            target = (target.detach().cpu().numpy()*639).astype("int32")
            
            # Update Progess Bar Status
            val_bar.set_description(desc=f"[Test] < Updating Results >")
            
        # Result Visualization
        for i in range(pred.shape[0]) :
            plt.clf()
            plt.figure(figsize=(10, 5))
            plt.plot(list(input[i,:,0])+list(pred[i,:,0]), list(input[i,:,1])+list(pred[i,:,1]), "r", label="Prediction")
            plt.plot(list(input[i,:,0])+list(target[i,:,0]), list(input[i,:,1])+list(target[i,:,1]), "g", label="Ground-Truth")
            plt.xlabel("Local X Coordinate")
            plt.ylabel("Local Y Coordinate")
            plt.title("Trajectory Tracking Prediction")
            plt.legend(loc="best")
            plt.savefig(f"{graph_dir}/trajectory_index_{index}_batch_{i}.png")
            plt.close()
            

if __name__ == "__main__" :
   # Read All Arguments
    opt = read_all_arguments()
    
    # Execute Main Function
    main(opt)