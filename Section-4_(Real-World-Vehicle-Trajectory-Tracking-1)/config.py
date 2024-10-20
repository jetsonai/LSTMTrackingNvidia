import argparse


def read_all_arguments() :
    # Create Argument Parser Instance
    parser = argparse.ArgumentParser()
    
    # Set Dataset Arguments
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.9)
    parser.add_argument("--target-frame", type=int, default=1)
    
    # Set Training Arguments (Hyperparameter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--total-epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay-rate", type=float, default=1e-2)
    parser.add_argument("--loss-function", type=str, default="L1", choices=["L1", "L2"])
    
    # Set Model Aruguments
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--hid-channels", type=int, default=256)
    parser.add_argument("--out-channels", type=int, default=2)
    parser.add_argument("--num-layer", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.1)
    
    # Parse Arguments
    opt = parser.parse_args()
    
    return opt