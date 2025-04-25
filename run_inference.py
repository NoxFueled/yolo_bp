import cv2
import time
from ultralytics import YOLO
import sys

def run_inference(model_name, video_input, video_output):
    
    model = YOLO(model_name)  

    
    cap = cv2.VideoCapture(video_input)

    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Video properties
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 'avc1' is the codec for H.264
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))

    # FPS kalkulacia
    prev_time = 0

    while True:
        # NAcitanie frame-u
        ret, frame = cap.read()
        if not ret:
            print("Video processing complete.")
            break

        
        current_time = time.time()

        # Calculate FPS
        frame_fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Inference
        results = model.predict(source=frame, show=False, conf=0.5)

        # Annotated frame s detekciou
        annotated_frame = results[0].plot()

        # FPS Overlay
        cv2.putText(
            annotated_frame,
            f"FPS: {frame_fps:.2f}",
            (10, 50),  # Position (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # Font scale
            (0, 255, 0),  # Color (BGR)
            2,  # Thickness
            cv2.LINE_AA,
        )

        # Write the frame to the output video
        out.write(annotated_frame)

    
    cap.release()
    out.release()  

if __name__ == "__main__":
    model_file = sys.argv[1]
    video_in = sys.argv[2]
    video_out = sys.argv[3]
    run_inference(model_file, video_in, video_out)

