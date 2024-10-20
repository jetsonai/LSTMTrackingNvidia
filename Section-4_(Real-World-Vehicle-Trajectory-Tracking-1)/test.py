from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

from dataset import get_dataloader
from model import LSTM
from utils import fix_seed, average_meter
from config import read_all_arguments


def main(opt) :
    # Fix Seed
    fix_seed(opt.seed)
    
    # Load Training Dataset
    _, _, test_loader, dataset = get_dataloader(opt, "dataset/HCMC-vehicle-dataset.csv")
    
    # Create Model Instance
    model = LSTM(opt).eval()
    
    # Load Pretraind Weight
    model.load_state_dict(torch.load(f"ckpt/lstm/{opt.loss_function}/best.pth"), strict=True)
    
    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type : {device}")
    
    # Assign Device
    model = model.to(device)
    
    # Retrieve Stats
    max = dataset.max
    
    # Create Directory
    graph_dir = f"result/test/{opt.loss_function}"
    makedirs(graph_dir, exist_ok=True)

    # Start Test Phase
    test_bar = tqdm(test_loader)
    
    # Create Average Meter Instance
    test_mae_loss, test_mse_loss = average_meter(), average_meter()
    
    # Start Test Phase
    for index, data in enumerate(test_bar) :
        # Load Dataset
        input, target = data
        
        # Assign Device
        input, target = input.to(device), target.to(device)
        
        with torch.no_grad() :
            # Get Prediction
            pred = model(input)
            
            # Affine Transformation
            input = (input.detach().cpu().numpy()*max).astype("int32")
            
            # Affine Transformation
            pred = pred[:,-opt.target_frame:,:]
            pred = (pred.detach().cpu().numpy()*max).astype("int32")

            # Affine Transformation
            target = (target.detach().cpu().numpy()*max).astype("int32")
            
            # Compute Loss
            test_mae_loss.update(np.abs(pred-target).mean(), target.shape[0])
            test_mse_loss.update(np.power(pred-target, 2).mean(), target.shape[0])

            # Update Progess Bar Status
            test_bar.set_description(desc=f"[Test] {opt.loss_function}-Model < MAE Loss : {test_mae_loss.avg:.4f} | MSE Loss : {test_mse_loss.avg:.4f} >")
    
    # Result Visualization
    for i in range(pred.shape[0]) :
        plt.clf()
        plt.figure(figsize=(10, 5))
        
        pred_x, pred_y = [], []
        target_x, target_y = [], []
        
        for j in range(input.shape[1]//2) :
            pred_x.append(input[i,j*2,0]), target_x.append(input[i,j*2,0])
            pred_y.append(input[i,j*2+1,0]), target_y.append(input[i,j*2+1,0])
        
        pred_x.append(pred[i,0,0]), target_x.append(target[i,0,0])
        pred_y.append(pred[i,0,1]), target_y.append(target[i,0,1])
        
        plt.plot(pred_x, pred_y, "r", label="Prediction")
        plt.plot(target_x, target_y, "g", label="Ground-Truth")
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