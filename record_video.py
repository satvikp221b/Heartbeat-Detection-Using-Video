import cv2
import time

def capture_video(video_path='output.mp4', record_time=15):
    """Capture the video from webcam for the time specified in seconds and store it"""
    cap = cv2.VideoCapture(0)

    # Check for camera cap
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set video frame and FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30  

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
    time.sleep(5)
    
    start_time = time.time()

    print(f"Recording video for {record_time} seconds...")
    while True:
        ret, frame = cap.read()

        
        if not ret:
            print("Failed to capture frame.")
            break

        # Write the frame to the output file
        out.write(frame)

        # Display the frame 
        cv2.imshow('Recording', frame)

        # Record for record_time seconds only
        if time.time() - start_time >= record_time:
            break

        # Stop Recording if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {video_path}")

# Record the video
capture_video(video_path='videos/input_whitelight.mp4', record_time=10)
