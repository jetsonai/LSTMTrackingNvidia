from os import makedirs

import matplotlib.pyplot as plt
import scipy

import torch

from tqdm import tqdm

from dataset import get_dataloader
from model import LSTM
from utils import fix_seed
from config import read_all_arguments


def main(opt) :
    # Fix Seed
    fix_seed(opt.seed)
    
    # Load Training Dataset
    _, _, test_loader, dataset = get_dataloader(opt, "dataset/WholeVdata2.csv")
    
    # Create Model Instance
    model = LSTM(opt).eval()
    
    # Load Pretraind Weight
    model.load_state_dict(torch.load(f"ckpt/lstm/best.pth"), strict=True)
    
    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type : {device}")
    
    # Assign Device
    model = model.to(device)
    
    # Retrieve Stats
    std = dataset.std[:4].to(device)
    mn = dataset.mn[:4].to(device)
    rg = dataset.range[:4].to(device)
    
    # Create Directory
    graph_dir = "result/test"
    makedirs(graph_dir, exist_ok=True)

    # Start Test Phase
    test_bar = tqdm(test_loader)
    
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
            pred = (pred*(rg*std) + mn).detach().cpu().numpy()
            pred = scipy.signal.savgol_filter(pred, window_length=5, polyorder=2,axis=1)

            # Affine Transformation
            target = (target*(rg*std)+mn).detach().cpu().numpy()
            
            # Replace Obeservation
            pred[:,:-opt.target_frame,:] = target[:,:-opt.target_frame,:]
            
            # Update Progess Bar Status
            test_bar.set_description(desc=f"[Test] < Updating Results >")
    
        # Result Visualization
        for i in range(pred.shape[0]) :
            plt.clf()
            plt.figure(figsize=(10, 5))
            plt.plot(pred[i,:,2], pred[i,:,3], "r", label="Prediction")
            plt.plot(target[i,:,2], target[i,:,3], "g", label="Ground-Truth")
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