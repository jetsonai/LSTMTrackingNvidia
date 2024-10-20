from os import makedirs
from random import shuffle

from collections import defaultdict

import cv2
import pandas as pd

from ultralytics import YOLO

from config import read_all_arguments
from utils import fix_seed


def main(opt) :
    # Load Pretrained YOLOv8 Model Weight
    yolo = YOLO("ckpt/yolov8/yolov8n.pt")
    yolo.info()
    
    # Load Video
    video_path = "dataset/video/train_val.avi"
    cap = cv2.VideoCapture(video_path)
    
    # Create Dictionary & List Instance
    track_history = defaultdict(lambda: [])
    data = []
    
    while cap.isOpened() :
        # Retrieve Frame
        ret, frame = cap.read()
        
        if ret :
            # Get Object Tracking Results
            results = yolo.track(frame, persist=True, imgsz=opt.video_size, conf=opt.conf, iou=opt.iou)
            
            # Get the Boxes and Track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()

            # Plot the tracks
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids) :
                if cls_id == 2 or cls_id == 5 or cls_id == 7 : # Car / Truck / Bus
                    x, y, w, h = box # Bounding Box Info
                    track = track_history[track_id]
                    x, y = float(x), float(y)
                    track.append((x, y)) # (x, y) Center Point
                    
                    if len(track) > (opt.input_frame + opt.target_frame) : # Input Frames + Target Frames
                        data.append(track) # Add Data
                        track.pop(0)
        else:
            break
    
    # Fix Seed
    fix_seed(opt.seed)
    
    # Shuffle Data
    shuffle(data)
    
    # Compute Training Dataset Size
    train_len = int(len(data)*opt.train_val_ratio)
    
    # Split Training & Validation Dataset
    train_df, val_df = pd.DataFrame(data=data[:train_len]), pd.DataFrame(data=data[train_len:])
    
    # Replace Column Names
    col_name = {}
    for i in range(opt.input_frame + opt.target_frame) :
        col_name[i] = f"frame_{i}"

    # Create Directory
    save_dir = "dataset/csv"
    makedirs(save_dir, exist_ok=True)
    
    # Save Training & Validation Dataset
    train_df, val_df = train_df.rename(columns=col_name), val_df.rename(columns=col_name)
    train_df.to_csv(f"{save_dir}/train_dataset.csv", index=False), val_df.to_csv(f"{save_dir}/val_dataset.csv", index=False)


if __name__ == "__main__" :
    # Read All Arguments
    opt = read_all_arguments()
    
    # Execute Main Function
    main(opt)