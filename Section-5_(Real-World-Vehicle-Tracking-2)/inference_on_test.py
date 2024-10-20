from os import makedirs
from collections import defaultdict

import cv2
import numpy as np

import torch

from ultralytics import YOLO
from model import LSTM
from config import read_all_arguments


def main(opt) :
    # Load Pretrained YOLOv8 Model Weight
    yolo = YOLO("ckpt/yolov8/yolov8n.pt")
    yolo.info()
    
    # Load Video
    video_path = "dataset/video/test.avi"
    cap = cv2.VideoCapture(video_path)
    
    # Create Directory for Saving Results
    save_dir = "result/lstm"
    makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(f"{save_dir}/tracking_prediction.avi", fourcc, cap.get(cv2.CAP_PROP_FPS), (opt.video_size, opt.video_size))
    
    # Create LSTM Model Instance
    lstm = LSTM(opt).eval()
    
    # Load Pretrained LSTM Model Weight
    weights = torch.load("ckpt/lstm/latest.pth")
    lstm.load_state_dict(weights, strict=True)
    
    # Determine Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Type : {device}")
    
    # Assign Device
    lstm = lstm.to(device)
    
    # Create Dictionary Instance
    track_history = defaultdict(lambda: [])
    lstm_track_history = defaultdict(lambda: [])
    
    with torch.no_grad() :
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

                # Visualize the Results on the Frame
                annotated_frame = results[0].plot()

                # Plot the tracks
                for box, track_id, cls_id in zip(boxes, track_ids, cls_ids) :
                    if cls_id == 2 or cls_id == 5 or cls_id == 7 : # Car / Truck / Bus
                        x, y, w, h = box # Bounding Box Info
                        track = track_history[track_id]
                        lstm_track = lstm_track_history[track_id]
                        track.append((float(x), float(y))) # (x, y) Center Point
                        if len(track) > opt.input_frame :
                            track.pop(0)
                            input = np.hstack(track).astype(np.int32).reshape((1, -1, 2)) # Get Input Data
                            input = torch.tensor(input).to(device)/639 # Min-Max Norm Input Data
                            pred = lstm(input)[:,-opt.target_frame:,:].clamp(0,1).cpu().detach().numpy().reshape(-1, 2)*639 # Inference & Affine Prediction
                            for i in range(pred.shape[0]) :
                                lstm_track.append((float(pred[i][0]), float(pred[i][1]))) # Add Predictions

                        # Draw the Tracking Lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
                        
                        # Draw the Predicted Tracking Lines
                        if len(lstm_track) == opt.target_frame :
                            pred_points = np.hstack(lstm_track).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(annotated_frame, [pred_points], isClosed=False, color=(255, 0, 0), thickness=2)
                        lstm_track_history[track_id] = []
                
                # Show Object Tracking Results
                out.write(annotated_frame)
                cv2.imshow("YOLOv8 Tracking with LSTM", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
    
    # Destroy UI
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__" :
    # Read All Arguments
    opt = read_all_arguments()
    
    # Execute Main Function
    main(opt)