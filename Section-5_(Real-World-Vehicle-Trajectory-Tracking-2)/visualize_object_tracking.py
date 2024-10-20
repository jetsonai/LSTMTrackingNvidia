from os import makedirs

from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

from config import read_all_arguments


def main(opt) :
    # Load Pretrained YOLOv8 Model Weight
    yolo = YOLO("ckpt/yolov8/yolov8n.pt")
    yolo.info()
    
    # Load Video
    video_path = "dataset/video/train_val.avi"
    cap = cv2.VideoCapture(video_path)
    
    # Create Directory for Saving Results
    save_dir = "result/object_tracking"
    makedirs(save_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(f"{save_dir}/object_tracking.avi", fourcc, cap.get(cv2.CAP_PROP_FPS), (opt.video_size, opt.video_size))
    
    # Create Dictionary for Saving Trajectory
    track_history = defaultdict(lambda: [])
    
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

            # Plot the Tracks
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids) :
                if cls_id == 2 or cls_id == 5 or cls_id == 7 : # Car / Truck / Bus
                    x, y, w, h = box # Bounding Box Info
                    track = track_history[track_id]
                    track.append((float(x), float(y))) # (x, y) Center Point
                    if len(track) > opt.input_frame : # Input Frames
                        track.pop(0)

                    # Draw the Tracking Lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=2)
            
            # Show Object Tracking Results
            out.write(annotated_frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q") :
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