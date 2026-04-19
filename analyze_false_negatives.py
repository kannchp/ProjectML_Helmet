"""
Analyze Misclassified Images - Find False Negatives
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def analyze_misclassified():
    """Find and visualize false negatives"""
    
    # Load best model
    model_path = r'C:\Users\title\Downloads\Project_ML\runs\detect\runs\detect\helmet_detection_v16_map=0.88_map95=0.67\weights\best.pt'
    model = YOLO(model_path)
    
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    
    val_imgs_dir = 'dataset/valid/images'
    val_labels_dir = 'dataset/valid/labels'
    
    false_negatives = []
    low_confidence = []
    
    print("🔍 Analyzing validation set for misclassifications...")
    
    for img_file in sorted(os.listdir(val_imgs_dir))[:50]:  # Check first 50
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
        gt_classes = [box['class'] for box in gt_boxes]
        
        # Get predictions
        results = model.predict(img_path, conf=0.3, verbose=False)
        pred_classes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                pred_classes.append(int(box.cls[0].item()))
        
        # Check for false negatives (class not detected)
        for gt_class in gt_classes:
            if gt_class not in pred_classes:
                false_negatives.append({
                    'image': img_file,
                    'path': img_path,
                    'missing_class': class_names[gt_class],
                    'gt_classes': [class_names[c] for c in gt_classes],
                    'pred_classes': [class_names[c] for c in pred_classes]
                })
        
        # Check for low confidence detections
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                conf = float(box.conf[0].item())
                if 0.3 <= conf < 0.5:
                    low_confidence.append({
                        'image': img_file,
                        'path': img_path,
                        'class': class_names[int(box.cls[0].item())],
                        'confidence': conf
                    })
    
    # Print results
    print("\n" + "="*70)
    print("FALSE NEGATIVES (Model missed these classes)")
    print("="*70)
    print(f"\nTotal false negatives: {len(false_negatives)}\n")
    
    for i, fn in enumerate(false_negatives[:10]):  # Show top 10
        print(f"{i+1}. {fn['image']}")
        print(f"   ❌ Missed:      {fn['missing_class']}")
        print(f"   ✓ Ground Truth: {fn['gt_classes']}")
        print(f"   🔍 Predicted:   {fn['pred_classes']}")
        print()
    
    print("\n" + "="*70)
    print("LOW CONFIDENCE DETECTIONS (Uncertain)")
    print("="*70)
    print(f"\nTotal low confidence: {len(low_confidence)}\n")
    
    for i, lc in enumerate(low_confidence[:10]):  # Show top 10
        print(f"{i+1}. {lc['image']}")
        print(f"   Class:      {lc['class']}")
        print(f"   Confidence: {lc['confidence']:.2f}")
        print()
    
    # Visualize top false negative
    if false_negatives:
        print("\n" + "="*70)
        print("VISUALIZING FALSE NEGATIVE EXAMPLES")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('False Negative Examples (Model Missed These)', fontsize=16)
        
        for idx, fn in enumerate(false_negatives[:4]):
            ax = axes[idx // 2, idx % 2]
            
            image = cv2.imread(fn['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Draw ground truth boxes
            label_file = fn['path'].replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            gt_boxes = load_annotations(label_file)
            
            for box in gt_boxes:
                x1, y1, x2, y2 = denormalize_box(box, image.shape[1], image.shape[0])
                color = (0, 255, 0) if class_names[box['class']] in fn['pred_classes'] else (255, 0, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_names[box['class']]}"
                cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            ax.imshow(image)
            ax.set_title(f"{fn['image']}\nMissed: {fn['missing_class']}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('false_negatives_analysis.jpg', dpi=150, bbox_inches='tight')
        print("✓ Saved visualization to: false_negatives_analysis.jpg")
        plt.close()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"\nFalse Negatives (missed detections): {len(false_negatives)}")
    print(f"Low Confidence Detections:          {len(low_confidence)}")
    
    if false_negatives:
        missing_classes = {}
        for fn in false_negatives:
            cls = fn['missing_class']
            missing_classes[cls] = missing_classes.get(cls, 0) + 1
        
        print(f"\nMost missed class:")
        for cls, count in sorted(missing_classes.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cls}: {count} times")

if __name__ == "__main__":
    analyze_misclassified()
