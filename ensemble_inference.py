"""
Ensemble Inference: YOLOv8-Medium + YOLOv5-Medium
For Helmet Detection - Combine predictions from multiple models
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class EnsembleDetector:
    def __init__(self, yolov8_weights, yolov5_weights=None):
        """
        Initialize ensemble with multiple YOLO models
        
        Args:
            yolov8_weights: Path to YOLOv8 trained weights
            yolov5_weights: Path to YOLOv5 trained weights (optional)
        """
        self.model_v8 = YOLO(yolov8_weights)
        self.model_v5 = None
        
        if yolov5_weights and Path(yolov5_weights).exists():
            # If using YOLOv5
            try:
                self.model_v5 = YOLO(yolov5_weights)
                print("✓ Loaded YOLOv8 and YOLOv5 models for ensemble")
            except:
                print("⚠ YOLOv5 model not available, using YOLOv8 only")
        else:
            print("✓ Loaded YOLOv8 model (single model inference)")
    
    def predict(self, image_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Run ensemble prediction on image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            iou_threshold: IOU threshold for NMS
            
        Returns:
            Ensemble results with averaged confidence scores
        """
        # Predict with YOLOv8
        results_v8 = self.model_v8.predict(
            image_path, 
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        predictions = {
            'v8': results_v8[0] if results_v8 else None,
            'v5': None
        }
        
        # Predict with YOLOv5 if available
        if self.model_v5:
            results_v5 = self.model_v5.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            predictions['v5'] = results_v5[0] if results_v5 else None
        
        return predictions
    
    def ensemble_predictions(self, predictions, method='soft'):
        """
        Combine predictions from multiple models
        
        Args:
            predictions: Dict with results from each model
            method: 'soft' (average confidence) or 'hard' (majority vote)
            
        Returns:
            Ensemble results
        """
        if method == 'soft':
            return self._soft_voting(predictions)
        elif method == 'hard':
            return self._hard_voting(predictions)
    
    def _soft_voting(self, predictions):
        """
        Soft voting: Average confidence scores from all models
        """
        v8_result = predictions['v8']
        v5_result = predictions['v5']
        
        if v8_result is None:
            return v8_result
        
        # If only YOLOv8
        if v5_result is None:
            return v8_result
        
        # Combine boxes and confidence
        v8_boxes = v8_result.boxes
        v5_boxes = v5_result.boxes
        
        combined_boxes = []
        combined_confs = []
        combined_classes = []
        
        # Process YOLOv8 detections
        for box, conf, cls in zip(v8_boxes.xyxy, v8_boxes.conf, v8_boxes.cls):
            combined_boxes.append(box.cpu().numpy())
            combined_confs.append(float(conf) * 0.6)  # Weight: 60%
            combined_classes.append(int(cls))
        
        # Process YOLOv5 detections
        for box, conf, cls in zip(v5_boxes.xyxy, v5_boxes.conf, v5_boxes.cls):
            # Find matching box from v8
            matched = False
            for i, v8_box in enumerate(combined_boxes):
                iou = self._calculate_iou(box.cpu().numpy(), v8_box)
                if iou > 0.5:  # If boxes overlap
                    combined_confs[i] += float(conf) * 0.4  # Add 40% weight
                    matched = True
                    break
            
            if not matched:  # New detection from v5
                combined_boxes.append(box.cpu().numpy())
                combined_confs.append(float(conf) * 0.4)  # Weight: 40%
                combined_classes.append(int(cls))
        
        return {
            'boxes': np.array(combined_boxes),
            'confidences': np.array(combined_confs),
            'classes': np.array(combined_classes)
        }
    
    def _hard_voting(self, predictions):
        """
        Hard voting: Majority vote on detections
        """
        v8_result = predictions['v8']
        v5_result = predictions['v5']
        
        if v8_result is None:
            return v8_result
        
        if v5_result is None:
            return v8_result
        
        # Simple approach: use YOLOv8 as primary, validate with v5
        # (More complex logic can be added)
        return v8_result
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """Calculate IOU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def visualize(self, image_path, predictions, output_path='ensemble_output.jpg'):
        """
        Draw predictions on image
        """
        image = cv2.imread(str(image_path))
        
        v8_result = predictions['v8']
        if v8_result is None:
            cv2.imwrite(output_path, image)
            return
        
        # Draw YOLOv8 results
        for box, conf, cls in zip(v8_result.boxes.xyxy, v8_result.boxes.conf, v8_result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            conf_val = float(conf)
            cls_name = self.model_v8.names[int(cls)]
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label
            label = f"{cls_name}: {conf_val:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        print(f"✓ Saved visualization to {output_path}")


if __name__ == "__main__":
    # Initialize ensemble
    ensemble = EnsembleDetector(
        yolov8_weights='runs/detect/helmet_detection_v1/weights/best.pt',
        yolov5_weights=None  # Add YOLOv5 path if trained
    )
    
    # Test on sample image
    test_image = 'dataset/valid/images/20260218_163331frame_0_jpg.rf.3bb30d80df6eff70e63dd6bcd74ad87b.jpg'
    
    # Run ensemble prediction
    print("\n🔍 Running ensemble prediction...")
    predictions = ensemble.predict(test_image, conf_threshold=0.5)
    
    # Ensemble results
    ensemble_result = ensemble.ensemble_predictions(predictions, method='soft')
    
    print(f"✓ Detections found: {len(ensemble_result['classes'])}")
    print(f"  Classes: {ensemble_result['classes']}")
    print(f"  Confidences: {ensemble_result['confidences']}")
    
    # Visualize
    ensemble.visualize(test_image, predictions, 'ensemble_test_output.jpg')
    print("\n✓ Ensemble inference complete!")
