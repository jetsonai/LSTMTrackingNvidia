import argparse


def read_all_arguments() :
    # Create Argument Parser Instance
    parser = argparse.ArgumentParser()
    
    # Set Dataset Arguments
    parser.add_argument("--video_size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.05)
    parser.add_argument("--input-frame", type=int, default=40)
    parser.add_argument("--target-frame", type=int, default=10)
    parser.add_argument("--train-val-ratio", type=float, default=0.9)
    
    # Set Training Arguments (Hyperparameter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay-rate", type=float, default=1e-2)
    
    # Set Model Aruguments
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--hid-channels", type=int, default=256)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--num-layer", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.1)
    
    # Parse Arguments
    opt = parser.parse_args()
    
    return opt