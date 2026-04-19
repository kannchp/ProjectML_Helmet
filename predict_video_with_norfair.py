import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Tracker, Detection
import os

# --- CONFIG ---
MODEL_PATH = "D:\\Project_ML\\runs\\detect\\runs\\detect\\basemodel.pt_finetune_v213\\weights\\best.pt"
VIDEO_PATH = 'D:\\projectML\\ตัดมาใช้\\testnohelmet5.mp4'
OUTPUT_PATH = 'oldlocation_output_norfair_tracked_modelv213_nohelmet5test.mp4'
CONF_THRES = 0.4
IOU_THRES = 0.25

TRACKER_TYPE = 'norfair'
CLASS_NAMES = ['helmet', 'motorcycle', 'no_helmet']
COLORS = {
    0: (0, 255, 0),      # Helmet - Green
    1: (255, 0, 0),      # Motorcycle - Blue
    2: (0, 0, 255),      # No helmet - Red
}

# --- LOAD MODEL ---
print(f"Loading model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# --- NORFAIR TRACKER ---
tracker = Tracker(
    distance_function="euclidean",
    distance_threshold=150,
    initialization_delay=1,
    hit_counter_max=3,
    
)

# --- VIDEO IO ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
print(f"\n📹 Processing video with Norfair Tracking...")
print(f"   Input: {VIDEO_PATH}")
print(f"   Output: {OUTPUT_PATH}")
print(f"   Resolution: {width}x{height}")
print(f"   FPS: {fps:.2f}")
print(f"   Total frames: {total_frames}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- DETECT with YOLO ---
        results = model.predict(
            frame, 
            conf=CONF_THRES, 
            iou=IOU_THRES, 
            verbose=False,
            agnostic_nms=True,
        )
        
        # Extract detections
        detections = []
        detection_classes = {}  # Store class info for each detection
        detection_scores = {}
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box
                
                # Create center point and bounding box points for Norfair
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Use 4 corner points for tracking
                points = np.array([
                    [x1, y1],
                    [x2, y1],
                    [x1, y2],
                    [x2, y2]
                ])
                
                detection = Detection(
                    points=points,
                    scores=np.array([conf] * 4),  # Score for each point
                    data={'box': [x1, y1, x2, y2], 'conf': conf, 'cls': cls_id}
                )
                detections.append(detection)
                detection_classes[i] = cls_id
                detection_scores[i] = conf
        
        # --- TRACK ---
        tracked_objects = tracker.update(detections=detections)
        
        # --- DRAW RESULTS ---
        frame_viz = frame.copy()
        
        for i, obj in enumerate(tracked_objects):
            if obj.last_detection is None:
                continue
            if obj.age < 3:
                continue
            # Get box from detection
            if 'box' in obj.last_detection.data:
                x1, y1, x2, y2 = map(int, obj.last_detection.data['box'])
                conf = obj.last_detection.data['conf']
                cls_id = obj.last_detection.data['cls']
            else:
                continue
            
            # Skip if box is completely outside frame
            if x2 <= 0 or x1 >= width or y2 <= 0 or y1 >= height:
                continue
            
            # Clamp coordinates to frame bounds
            x1_clamped = max(0, min(x1, width - 1))
            y1_clamped = max(0, min(y1, height - 1))
            x2_clamped = max(0, min(x2, width - 1))
            y2_clamped = max(0, min(y2, height - 1))
            
            if x1_clamped >= x2_clamped or y1_clamped >= y2_clamped:
                continue
            
            # Calculate visible area ratio
            original_area = (x2 - x1) * (y2 - y1)
            visible_area = (x2_clamped - x1_clamped) * (y2_clamped - y1_clamped)
            visible_ratio = visible_area / max(original_area, 1)
            
            # Skip if less than 30% of box is visible in frame
            if visible_ratio < 0.3:
                continue
            
            x1, y1, x2, y2 = x1_clamped, y1_clamped, x2_clamped, y2_clamped
            
            # Get color for this class
            color = COLORS.get(int(cls_id), (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # Draw corner markers
            corner_len = 20
            cv2.line(frame_viz, (x1, y1), (x1+corner_len, y1), color, 3)
            cv2.line(frame_viz, (x1, y1), (x1, y1+corner_len), color, 3)
            cv2.line(frame_viz, (x2, y1), (x2-corner_len, y1), color, 3)
            cv2.line(frame_viz, (x2, y1), (x2, y1+corner_len), color, 3)
            cv2.line(frame_viz, (x1, y2), (x1+corner_len, y2), color, 3)
            cv2.line(frame_viz, (x1, y2), (x1, y2-corner_len), color, 3)
            cv2.line(frame_viz, (x2, y2), (x2-corner_len, y2), color, 3)
            cv2.line(frame_viz, (x2, y2), (x2, y2-corner_len), color, 3)
            
            # Draw track ID and class name
            class_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else f"Class {int(cls_id)}"
            label = f"ID: {obj.id} | {class_name} ({conf:.2f})"
            
            # Text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_h = text_size[1] + 10
            text_w = text_size[0] + 10
            
            bg_x1 = max(0, x1 - 5)
            bg_y1 = max(0, y1 - text_h - 5)
            bg_x2 = min(width, x1 + text_w + 5)
            bg_y2 = max(y1, y1)
            
            cv2.rectangle(frame_viz, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.rectangle(frame_viz, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
            
            cv2.putText(frame_viz, label, (bg_x1+5, bg_y1+text_size[1]+3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter and info
        cv2.putText(frame_viz, f"Frame: {frame_count} | Tracks: {len(tracked_objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_viz, f"Conf: {CONF_THRES} | Tracker: {TRACKER_TYPE.upper()}", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        out.write(frame_viz)
        cv2.imshow('Norfair Tracking', frame_viz)
        
        if frame_count % 50 == 0:
            print(f"   Processed {frame_count} frames... ({len(tracked_objects)} active tracks)")
        
        frame_count += 1
        
        # Press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠ Interrupted by user")
            break
    
    print(f"\n✅ Done! Processed {frame_count} frames")
    print(f"💾 Output saved to {OUTPUT_PATH}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
