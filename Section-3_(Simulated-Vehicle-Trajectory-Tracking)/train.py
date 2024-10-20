from os import makedirs

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from tqdm import tqdm

from dataset import get_dataloader
from model import LSTM
from utils import fix_seed, average_meter
from config import read_all_arguments


def main(opt) :
    # Fix Seed
    fix_seed(opt.seed)
    
    # Load Training Dataset
    train_loader, val_loader, _, _ = get_dataloader(opt, "dataset/WholeVdata2.csv")
    
    # Fix Seed
    fix_seed(opt.seed)
    
    # Create Model Instance
    model = LSTM(opt)
    
    # Compute Number of Parameters
    num_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# Parameters : {num_parameter:,}")
    
    # Create Optimizer Instance
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    # Create Scheduler Instance
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=opt.total_epoch*len(train_loader),
                                                     eta_min=opt.lr*opt.decay_rate)
    
    # Create Loss Function Instance
    criterion = nn.L1Loss()
    
    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type : {device}")
    
    # Assign Device
    model = model.to(device)
    
    # Create Average Meter Instance
    train_loss, val_loss = average_meter(), average_meter()
    
    # Create List Instance
    train_loss_list, val_loss_list = [], []
    
    # Create Directory
    ckpt_dir, graph_dir = "ckpt/lstm", "result/lstm"
    makedirs(ckpt_dir, exist_ok=True), makedirs(graph_dir, exist_ok=True)
    
    # Set Best loss
    best_loss = np.inf
    
    # Start Training
    for epoch in range(1, opt.total_epoch+1) :
        # Create TQDM Dataloader Instance
        train_bar = tqdm(train_loader)
        
        # Reset Average Meter Instance
        train_loss.reset()
        
        # Set Training Mode
        model.train()
        
        # Training Phase
        for data in train_bar :
            # Load Dataset
            input, target = data
            
            # Assign Device
            input, target = input.to(device), target.to(device)
            
            # Set Gradient to 0
            optimizer.zero_grad() 
            
            # Get Prediction
            pred = model(input)
            
            # Compute Loss
            loss = criterion(pred[:,-opt.target_frame:,2:4], target[:,-opt.target_frame:,2:4]) 
            
            # Back-Propagation
            loss.backward() 
            
            # Update Weight
            optimizer.step()
            
            # Update Learning Rate Scheduler
            scheduler.step()
            
            # Compute Averaged Loss
            train_loss.update(loss.detach().cpu().item(), opt.batch_size)
            
            # Update Progess Bar Status
            train_bar.set_description(desc=f"[{epoch}/{opt.total_epoch}] [Train] < Loss:{train_loss.avg:.4f} >")
        
        # Add Training Loss
        train_loss_list.append(train_loss.avg)
        
        # Create TQDM Dataloader Instance
        val_bar = tqdm(val_loader)
        
        # Reset Average Meter Instance
        val_loss.reset()
        
        # Set Validation Mode
        model.eval()
        
        # Validation Phase
        for data in val_bar :
            # Load Dataset
            input, target = data
            
            # Assign Device
            input, target = input.to(device), target.to(device)
            
            with torch.no_grad() :
                # Get Prediction
                pred = model(input)
                
            # Compute Loss
            loss = criterion(pred[:,-opt.target_frame:,2:4], target[:,-opt.target_frame:,2:4]) 
            
            # Compute Averaged Loss
            val_loss.update(loss.detach().cpu().item(), opt.batch_size) 
            
            # Update Progess Bar Status
            val_bar.set_description(desc=f"[{epoch}/{opt.total_epoch}] [Val] < Loss:{val_loss.avg:.4f} >")
            
        # Add Validation Loss
        val_loss_list.append(val_loss.avg)
        
        # Save Network
        if val_loss.avg < best_loss :
            best_loss = val_loss.avg
            torch.save(model.state_dict(), f"{ckpt_dir}/best.pth")
        torch.save(model.state_dict(), f"{ckpt_dir}/latest.pth")
        
        # Plot Training vs. Validation Loss Graph
        plt.clf()
        plt.plot(np.arange(epoch), train_loss_list, label="Training Loss")
        plt.plot(np.arange(epoch), val_loss_list, label="Validation Loss")
        plt.title("Loss (Training vs. Validation)")
        plt.xlabel("Epoch"), plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(f"{graph_dir}/loss.png")


if __name__ == "__main__" :
   # Read All Arguments
    opt = read_all_arguments()
    
    # Execute Main Function
    main(opt)