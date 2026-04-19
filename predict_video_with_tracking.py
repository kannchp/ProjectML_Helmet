import cv2
import numpy as np
from ultralytics import YOLO
from tracker import YOLOTrackerWithStability, visualize_tracks

# --- CONFIG ---
MODEL_PATH = 'runs/detect/runs/detect/helmet_detection_v17/weights/best.pt'
VIDEO_PATH = 'clip_test.mp4'
OUTPUT_PATH = 'output_tracked2.mp4'
CONF_THRES = 0.45  # Increase threshold to reduce false positives
IOU_THRES = 0.5   # Lower IOU for stricter NMS
NMS_THRESHOLD = 0.3  # Stricter NMS to remove more overlapping boxes
TRACKER_TYPE = 'botsort'
CLASS_NAMES = ['helmet', 'motorcycle', 'no_helmet']
COLORS = {
    0: (0, 255, 0),      # Helmet - Green
    1: (255, 0, 0),      # Motorcycle - Blue
    2: (0, 0, 255),      # No helmet - Red
}

# --- LOAD MODEL AND TRACKER ---
model = YOLO(MODEL_PATH)
tracker = YOLOTrackerWithStability(
    model, 
    tracker_type=TRACKER_TYPE, 
    persist=True,
    conf_threshold=CONF_THRES,
    class_smooth=True,
    detection_smooth=True,  # Enable detection smoothing
    max_missing_frames=3,   # Keep interpolating for up to 3 missing frames
    nms_threshold=NMS_THRESHOLD  # Enable NMS to remove duplicate detections
)

# --- VIDEO IO ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
print(f"\n📹 Processing video...")
print(f"   Input: {VIDEO_PATH}")
print(f"   Output: {OUTPUT_PATH}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track (with stability filtering)
        tracks = tracker.track(frame, iou=IOU_THRES)
        
        # Visualize with increased thickness for visibility
        frame_viz = visualize_tracks(frame, tracks, CLASS_NAMES, COLORS, thickness=3)
        
        # Add frame counter and info
        cv2.putText(frame_viz, f"Frame: {frame_count} | Tracks: {tracks['num_tracks']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_viz, f"Conf: {CONF_THRES} | NMS: {NMS_THRESHOLD} | Smoothing: ON | {TRACKER_TYPE.upper()}", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        out.write(frame_viz)
        cv2.imshow('Tracking', frame_viz)
        
        if frame_count % 50 == 0:
            print(f"   Processed {frame_count} frames... ({tracks['num_tracks']} active tracks)")
        
        frame_count += 1
        
        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⚠ Interrupted by user")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'\n✅ Done! Processed {frame_count} frames')
    print(f'💾 Output saved to {OUTPUT_PATH}')
