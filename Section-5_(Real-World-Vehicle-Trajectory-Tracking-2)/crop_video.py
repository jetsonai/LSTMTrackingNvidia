import cv2

from config import read_all_arguments


def main(opt) :
    # Read Video
    video_path = "dataset/video/sherbrooke_video.avi"
    cap = cv2.VideoCapture(video_path)
    
    # Get Video Meta-Data
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Print Video Meta-Data
    print(f"FPS : {fps}")
    print(f"Frame Count : {frame_count}")
    print(f"Video Size : {width} x {height}")
    
    # Save Video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    train_val_out = cv2.VideoWriter("dataset/video/train_val.avi", fourcc, fps, (opt.video_size, opt.video_size))
    test_out = cv2.VideoWriter("dataset/video/test.avi", fourcc, fps, (opt.video_size, opt.video_size))
    
    # Split Video
    frame_id = 0
    while cap.isOpened() :
        # Retrieve Frame
        ret, frame = cap.read()
        
        # Crop Video
        frame = frame[:,100:700,:]
        
        # Resize Video
        frame = cv2.resize(frame, (opt.video_size, opt.video_size))
        
        if ret and frame_id < frame_count :
            # Train-Valid. Section
            if frame_id < 3000 :
                train_val_out.write(frame)
                cv2.imshow("Train-Val Window", frame)
            
            # Test Section
            else :
                test_out.write(frame)
                cv2.imshow("Test Window", frame)
            
            # Count Frame ID
            frame_id += 1

            if cv2.waitKey(25) == ord('q') :
                break
        else :
            break
    
    # Destory UI
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__" :
    # Read All Arguments
    opt = read_all_arguments()
    
    # Execute Main Function
    main(opt)