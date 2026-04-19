"""
Tracker Module - Multi-object tracking for helmet detection
Supports ByteTrack, BoT-SORT, with stability and NMS filtering
"""
import cv2
import numpy as np
from pathlib import Path

class TrackingState:
    """
    Keep track of object states to ensure stability
    Prevents rapid class switching and low-confidence detections
    Includes detection smoothing for missing detections
    """
    def __init__(self, track_id, cls_id, conf, box, max_history=5, max_missing_frames=3):
        self.track_id = track_id
        self.class_history = [cls_id] * max_history
        self.conf_history = [conf] * max_history
        self.box_history = [box.copy()]  # Keep position history
        self.max_history = max_history
        self.frame_count = 0
        self.missing_frames = 0  # Counter for frames without detection
        self.max_missing_frames = max_missing_frames  # Max frames to interpolate
        self.last_valid_box = box.copy()
    
    def update(self, cls_id, conf, box):
        """Update with new detection"""
        self.class_history.pop(0)
        self.class_history.append(cls_id)
        self.conf_history.pop(0)
        self.conf_history.append(conf)
        self.box_history.append(box.copy())
        if len(self.box_history) > self.max_history:
            self.box_history.pop(0)
        self.last_valid_box = box.copy()
        self.missing_frames = 0
        self.frame_count += 1
    
    def miss_frame(self):
        """Called when object is not detected in current frame"""
        self.missing_frames += 1
    
    def interpolate_box(self):
        """Estimate box position based on motion history"""
        if len(self.box_history) < 2:
            return self.last_valid_box.copy()
        
        # Linear extrapolation from last 2 boxes
        prev_box = self.box_history[-1]
        prev_prev_box = self.box_history[-2] if len(self.box_history) >= 2 else self.box_history[-1]
        
        # Estimate velocity
        dx = prev_box[0] - prev_prev_box[0]
        dy = prev_box[1] - prev_prev_box[1]
        dw = prev_box[2] - prev_prev_box[2]
        dh = prev_box[3] - prev_prev_box[3]
        
        # Extrapolate
        new_box = prev_box.copy()
        new_box[0] += dx
        new_box[1] += dy
        new_box[2] += dw
        new_box[3] += dh
        
        return new_box
    
    def should_display(self):
        """Check if should still display this track"""
        return self.missing_frames <= self.max_missing_frames
    
    def get_stable_class(self):
        """Get most common class (majority voting)"""
        from collections import Counter
        counts = Counter(self.class_history)
        return counts.most_common(1)[0][0] if counts else self.class_history[-1]
    
    def get_avg_confidence(self):
        """Get average confidence"""
        return np.mean(self.conf_history)
    

def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove duplicate/overlapping boxes
    
    Args:
        boxes: Array of [x1, y1, x2, y2] coordinates
        scores: Confidence scores for each box
        iou_threshold: IOU threshold for suppression (default 0.5)
    
    Returns:
        indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=int)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by score descending
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IOU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)
        
        # Keep boxes with IOU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


class YOLOTrackerWithStability:
    """
    YOLO tracker with class stability, confidence filtering, and detection smoothing
    Reduces flickering and missing detections through interpolation
    """
    
    def __init__(self, model, tracker_type='botsort', persist=True, 
                 conf_threshold=0.35, class_smooth=True, 
                 detection_smooth=True, max_missing_frames=2,
                 nms_threshold=0.5, max_track_age=30):
        """
        Initialize tracker with stability, smoothing, and NMS
        
        Args:
            model: YOLO model instance
            tracker_type: 'botsort' or 'bytetrack'
            persist: Keep track IDs across frames
            conf_threshold: Filter detections below this confidence
            class_smooth: Enable class smoothing via majority voting
            detection_smooth: Enable detection smoothing via interpolation
            max_missing_frames: Max frames to interpolate when object is missing
            nms_threshold: IOU threshold for NMS (0-1, lower = more suppression)
        """
        self.model = model
        self.tracker_type = tracker_type
        self.persist = persist
        self.conf_threshold = conf_threshold
        self.class_smooth = class_smooth
        self.detection_smooth = detection_smooth
        self.max_missing_frames = max_missing_frames
        self.nms_threshold = nms_threshold
        self.max_track_age = max_track_age  # Max frames to keep track alive
        self.track_states = {}  # {track_id: TrackingState}
        
        print(f"✓ Initialized {tracker_type.upper()} tracker with stability")
        print(f"  - Persist: {persist}")
        print(f"  - Conf threshold: {conf_threshold}")
        print(f"  - Class smoothing: {class_smooth}")
        print(f"  - Detection smoothing: {detection_smooth}")
        if detection_smooth:
            print(f"  - Max missing frames: {max_missing_frames}")
        print(f"  - NMS threshold: {nms_threshold} (IOU-based suppression)")
    
    def track(self, frame, conf=None, iou=0.6, device=None):
        """
        Track objects in frame with stability and smoothing
        
        Returns:
            dict with tracking results (with stable classes and interpolated boxes)
        """
        if conf is None:
            conf = self.conf_threshold
        
        tracker_config = f'{self.tracker_type}.yaml'
        kwargs = dict(
            conf=conf,
            iou=iou,
            tracker=tracker_config,
            persist=self.persist,
            verbose=False,
        )
        if device is not None:
            kwargs["device"] = device
        
        results = self.model.track(frame, **kwargs)
        
        # Extract current detections
        current_detections = {}
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Filter by confidence threshold
            for box, track_id, conf_val, cls_id in zip(boxes, track_ids, confidences, classes):
                if conf_val >= self.conf_threshold:
                    current_detections[track_id] = {
                        'box': box,
                        'conf': conf_val,
                        'cls': cls_id
                    }
        
        # Update tracking states
        for track_id, det_info in current_detections.items():
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackingState(
                    track_id, 
                    det_info['cls'], 
                    det_info['conf'],
                    det_info['box'],
                    max_missing_frames=self.max_missing_frames
                )
            else:
                self.track_states[track_id].update(
                    det_info['cls'], 
                    det_info['conf'],
                    det_info['box']
                )
        
        # Handle missing detections (mark as missing but keep track)
        for track_id in list(self.track_states.keys()):
            if track_id not in current_detections:
                self.track_states[track_id].miss_frame()
        
        # Build output with smoothing
        output_boxes = []
        output_track_ids = []
        output_confidences = []
        output_classes = []
        
        for track_id, state in self.track_states.items():
            # Only display if within max missing frames
            if not state.should_display():
                continue
            
            # Get box (either current or interpolated)
            if track_id in current_detections:
                box = current_detections[track_id]['box']
                conf = current_detections[track_id]['conf']
            else:
                # Use interpolation for missing detections
                if self.detection_smooth:
                    box = state.interpolate_box()
                    conf = state.get_avg_confidence()
                else:
                    continue
            
            # Get stable class if enabled
            if self.class_smooth:
                cls_id = state.get_stable_class()
            else:
                cls_id = state.class_history[-1]
            
            output_boxes.append(box)
            output_track_ids.append(track_id)
            output_confidences.append(conf)
            output_classes.append(cls_id)
        
        # Apply NMS to remove duplicate/overlapping detections
        if len(output_boxes) > 0:
            boxes_array = np.array(output_boxes)
            confidences_array = np.array(output_confidences)
            
            # Apply NMS
            keep_indices = apply_nms(boxes_array, confidences_array, self.nms_threshold)
            
            # Filter outputs
            output_boxes = boxes_array[keep_indices].tolist()
            output_track_ids = [output_track_ids[i] for i in keep_indices]
            output_confidences = confidences_array[keep_indices].tolist()
            output_classes = [output_classes[i] for i in keep_indices]
        
        # Cleanup old tracks - remove if missing for too long OR too old
        self.track_states = {
            tid: state for tid, state in self.track_states.items() 
            if state.should_display() and state.frame_count < self.max_track_age
        }
        
        return {
            'boxes': np.array(output_boxes),
            'track_ids': np.array(output_track_ids),
            'confidences': np.array(output_confidences),
            'classes': np.array(output_classes),
            'num_tracks': len(output_track_ids)
        }


def visualize_tracks(frame, tracks, class_names=None, colors=None, thickness=3):
    """
    Visualize tracking results with bounding boxes and track IDs
    
    Args:
        frame: Input frame
        tracks: Tracking results from tracker
        class_names: List of class names ['helmet', 'motorcycle', 'no_helmet']
        colors: Dict of {class_id: (B,G,R)} or {track_id: (B,G,R)}
        thickness: Line thickness for boxes
    
    Returns:
        Annotated frame
    """
    frame_viz = frame.copy()
    
    if class_names is None:
        class_names = ['helmet', 'motorcycle', 'no_helmet']
    
    if colors is None:
        # Default colors by class (brighter colors)
        colors = {
            0: (0, 255, 0),      # Helmet - Green
            1: (255, 0, 0),      # Motorcycle - Blue
            2: (0, 0, 255),      # No helmet - Red
        }
    
    boxes = tracks['boxes']
    track_ids = tracks['track_ids']
    confidences = tracks['confidences']
    classes = tracks['classes']
    
    if len(boxes) == 0:
        return frame_viz
    
    for box, track_id, conf, cls_id in zip(boxes, track_ids, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        
        # Skip if completely outside frame
        if x2 < 0 or x1 >= frame_viz.shape[1] or y2 < 0 or y1 >= frame_viz.shape[0]:
            continue
        
        # Skip if box is too small
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width < 10 or box_height < 10:
            continue
        
        # Validate coordinates (clamp to frame)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_viz.shape[1], x2)
        y2 = min(frame_viz.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        # Get color for this class
        color = colors.get(int(cls_id), (255, 255, 255))
        
        # Draw outer box (thicker, for visibility)
        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), color, thickness)
        
        # Draw inner box (contrast)
        cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        # Draw corner markers for better visibility
        corner_len = 20
        cv2.line(frame_viz, (x1, y1), (x1+corner_len, y1), color, thickness)
        cv2.line(frame_viz, (x1, y1), (x1, y1+corner_len), color, thickness)
        cv2.line(frame_viz, (x2, y1), (x2-corner_len, y1), color, thickness)
        cv2.line(frame_viz, (x2, y1), (x2, y1+corner_len), color, thickness)
        cv2.line(frame_viz, (x1, y2), (x1+corner_len, y2), color, thickness)
        cv2.line(frame_viz, (x1, y2), (x1, y2-corner_len), color, thickness)
        cv2.line(frame_viz, (x2, y2), (x2-corner_len, y2), color, thickness)
        cv2.line(frame_viz, (x2, y2), (x2, y2-corner_len), color, thickness)
        
        # Draw track ID and class
        class_name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else f"Class {int(cls_id)}"
        label = f"ID: {int(track_id)} | {class_name} ({conf:.2f})"
        
        # Background for text (with padding)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_h = text_size[1] + 10
        text_w = text_size[0] + 10
        
        bg_x1 = max(0, x1 - 5)
        bg_y1 = max(0, y1 - text_h - 5)
        bg_x2 = min(frame_viz.shape[1], x1 + text_w + 5)
        bg_y2 = max(y1, y1)
        
        cv2.rectangle(frame_viz, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.rectangle(frame_viz, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
        
        cv2.putText(frame_viz, label, (bg_x1+5, bg_y1+text_size[1]+3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_viz


def test_tracking(model_path, video_path, output_path='output_tracked.mp4', 
                   conf_thres=0.25, iou_thres=0.6, tracker_type='botsort',
                   enable_smoothing=True):
    """
    Test tracking on video with smoothing
    
    Args:
        model_path: Path to YOLO model
        video_path: Input video path
        output_path: Output video path
        conf_thres: Confidence threshold
        iou_thres: IOU threshold
        tracker_type: 'botsort' or 'bytetrack'
        enable_smoothing: Enable detection smoothing
    """
    from ultralytics import YOLO
    
    print("="*70)
    print(f"Testing {tracker_type.upper()} Tracking with Detection Smoothing")
    print("="*70)
    
    # Load model
    model = YOLO(model_path)
    tracker = YOLOTrackerWithStability(
        model, 
        tracker_type=tracker_type, 
        persist=True,
        conf_threshold=conf_thres,
        class_smooth=True,
        detection_smooth=enable_smoothing,
        max_missing_frames=3
    )
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\n📹 Input: {video_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Total frames: {total_frames}")
    print(f"\n🎯 Detection Smoothing: {'✅ ON (interpolates missing detections)' if enable_smoothing else '❌ OFF'}")
    
    frame_idx = 0
    track_history = {}
    missing_frame_count = 0
    interpolated_frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            tracks = tracker.track(frame, conf=conf_thres, iou=iou_thres)
            
            # Record track history
            for track_id in tracks['track_ids']:
                if track_id not in track_history:
                    track_history[track_id] = {'first': frame_idx, 'last': frame_idx, 'count': 0}
                track_history[track_id]['last'] = frame_idx
                track_history[track_id]['count'] += 1
            
            # Visualize
            frame_viz = visualize_tracks(frame, tracks)
            
            # Add frame counter
            cv2.putText(frame_viz, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame_viz)
            
            if frame_idx % 50 == 0:
                print(f"  Processed frame {frame_idx}/{total_frames} ({frame_idx/fps:.1f}s) - "
                      f"Active tracks: {tracks['num_tracks']}")
            
            frame_idx += 1
    
    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user")
    
    finally:
        cap.release()
        out.release()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 Track Summary:")
    print("="*70)
    for track_id, info in sorted(track_history.items()):
        duration = (info['last'] - info['first']) / fps
        print(f"Track ID {track_id}: {info['count']:4d} frames | "
              f"{info['first']/fps:6.1f}s - {info['last']/fps:6.1f}s | "
              f"duration: {duration:.1f}s")
    
    print("\n" + "="*70)
    print(f"✅ Tracking completed!")
    print(f"💾 Output saved to: {output_path}")
    if enable_smoothing:
        print(f"📊 Detection Smoothing was ACTIVE (reduced flickering)")
    print("="*70)


if __name__ == "__main__":
    test_tracking(
        model_path='runs/detect/runs/detect/helmet_detection_v16_map=0.88_map95=0.67/weights/best.pt',
        video_path='clip_test.mp4',
        output_path='output_tracked.mp4',
        conf_thres=0.3,
        iou_thres=0.6,
        tracker_type='botsort'
    )
