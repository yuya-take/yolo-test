import cv2
from ultralytics import YOLO

def main():
    
    model = YOLO("yolo11n.pt")
    
    # set the input and output video paths
    input_video_path = "data/input_video.mp4"
    output_video_path = "data/output_video.mp4"
    
    # read the input video information
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"video information: fps={fps}, width={width}, height={height}")
    
    # create the output video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # read the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # rotate the frame
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        # detect objects in the frame
        result_list = model(frame)
        
        # draw the bounding boxes
        for result in result_list:
            frame = result.plot()
        
        # write the frame to the output video
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    out.release()
    
    print("Done!")

if __name__ == "__main__":
    main()
