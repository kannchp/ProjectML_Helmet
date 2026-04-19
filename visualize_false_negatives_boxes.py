"""
Visualize False Negatives with Ground Truth and Prediction Boxes
"""

import os
import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO

def load_annotations(label_file):
    """Load YOLO format annotations"""
    boxes = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id, x_center, y_center, width, height = map(float, parts[:5])
                    boxes.append({
                        'class': int(cls_id),
                        'x': x_center,
                        'y': y_center,
                        'w': width,
                        'h': height
                    })
    return boxes

def denormalize_box(box, img_width, img_height):
    """Convert normalized box to pixel coordinates"""
    x_center = box['x'] * img_width
    y_center = box['y'] * img_height
    width = box['w'] * img_width
    height = box['h'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

def draw_false_negatives():
    """Draw boxes on false negative images"""
    
    # Create output directory
    base_output = 'false_negatives_with_boxes'
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(f'{base_output}/missed_helmet', exist_ok=True)
    os.makedirs(f'{base_output}/missed_no_helmet', exist_ok=True)
    os.makedirs(f'{base_output}/missed_motorcycle', exist_ok=True)
    
    # Load best model
    model_path = r'C:\Users\title\Downloads\Project_ML\runs\detect\runs\detect\helmet_detection_v16_map=0.88_map95=0.67\weights\best.pt'
    model = YOLO(model_path)
    
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    class_colors = {
        'helmet': (0, 255, 0),      # Green
        'motorcycle': (255, 0, 0),  # Blue
        'no_helmet': (0, 0, 255)    # RedC:\Users\title\Downloads\Project_ML\runs\detect\runs\detect\helmet_detection_v16_map=0.88_map95=0.67
    }
    
    class_folders = {
        'helmet': f'{base_output}/missed_helmet',
        'motorcycle': f'{base_output}/missed_motorcycle',
        'no_helmet': f'{base_output}/missed_no_helmet'
    }
    
    val_imgs_dir = 'dataset/valid/images'
    val_labels_dir = 'dataset/valid/labels'
    
    false_negatives_by_class = {'helmet': set(), 'motorcycle': set(), 'no_helmet': set()}
    count = 0
    
    print("🎨 Drawing bounding boxes on false negative images...")
    
    for img_file in sorted(os.listdir(val_imgs_dir))[:100]:  # Process first 100
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(val_imgs_dir, img_file)
        label_file = os.path.join(val_labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # Get ground truth
        gt_boxes = load_annotations(label_file)
        gt_classes = set([box['class'] for box in gt_boxes])
        
        # Get predictions
        results = model.predict(img_path, conf=0.3, verbose=False)
        pred_classes = set()
        pred_boxes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                pred_classes.add(cls_id)
                pred_boxes.append({
                    'xyxy': box.xyxy[0].cpu().numpy(),
                    'conf': float(box.conf[0].item()),
                    'class': cls_id
                })
        
        # Find missing classes
        missing_classes = gt_classes - pred_classes
        
        if len(missing_classes) == 0:
            continue
        
        # Draw ground truth boxes (all)
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = denormalize_box(gt_box, width, height)
            cls_id = gt_box['class']
            cls_name = class_names[cls_id]
            color = class_colors[cls_name]
            
            # Green = detected, Red = missed
            if cls_id in missing_classes:
                color = (0, 0, 255)  # Red for missed
                thickness = 2
                label = f"✗ {cls_name}"
            else:
                color = (0, 255, 0)  # Green for detected
                thickness = 1
                label = f"✓ {cls_name}"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Add text background
            font_scale = 0.4
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                   font_scale, font_thickness)
            cv2.rectangle(image, (x1, y1-text_height-5), (x1+text_width+5, y1), color, -1)
            cv2.putText(image, label, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), font_thickness)
        
        # Draw prediction boxes
        for pred_box in pred_boxes:
            x1, y1, x2, y2 = map(int, pred_box['xyxy'])
            cls_name = class_names[pred_box['class']]
            color = class_colors[cls_name]
            conf = pred_box['conf']
            
            # Draw dashed border for predictions
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            label = f"{cls_name}: {conf:.2f}"
            font_scale = 0.35
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                           font_scale, 1)
            cv2.rectangle(image, (x1, y2), (x1+text_width+4, y2+text_height+4), color, -1)
            cv2.putText(image, label, (x1+2, y2+text_height+2), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), 1)
        
        # Add legend
        cv2.putText(image, "GREEN=Detected | RED=Missed | Color=Type",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save to appropriate folder
        for missing_cls in missing_classes:
            cls_name = class_names[missing_cls]
            false_negatives_by_class[cls_name].add(img_file)
            
            output_path = os.path.join(class_folders[cls_name], img_file)
            cv2.imwrite(output_path, image)
            count += 1
    
    # Print summary
    print("\n" + "="*70)
    print("FALSE NEGATIVES WITH BOUNDING BOXES")
    print("="*70)
    
    print(f"\n📁 Output folder: {base_output}/\n")
    
    for cls_name in ['helmet', 'no_helmet', 'motorcycle']:
        count_cls = len(false_negatives_by_class[cls_name])
        print(f"❌ Missed {cls_name:12} : {count_cls:3d} images")
        print(f"   Folder: {class_folders[cls_name]}/")
        if count_cls > 0 and count_cls <= 5:
            for img in list(false_negatives_by_class[cls_name]):
                print(f"      - {img}")
        elif count_cls > 5:
            for img in list(false_negatives_by_class[cls_name])[:3]:
                print(f"      - {img}")
            print(f"      ... and {count_cls - 3} more")
        print()
    
    print("="*70)
    print("📌 Legend:")
    print("   🟢 GREEN box    = Ground Truth (detected by model)")
    print("   🔴 RED box      = Ground Truth (MISSED by model)")
    print("   ⬜ DASHED box   = Model prediction")
    print("="*70)

if __name__ == "__main__":
    draw_false_negatives()
