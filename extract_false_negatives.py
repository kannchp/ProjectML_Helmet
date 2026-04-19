"""
Extract False Negative Images to Separate Folder
"""

import os
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
                    cls_id = int(float(parts[0]))
                    boxes.append(cls_id)
    return boxes

def extract_false_negatives():
    """Extract and organize false negative images"""
    
    # Create output directories
    base_output = 'false_negatives_analysis'
    os.makedirs(base_output, exist_ok=True)
    os.makedirs(f'{base_output}/missed_helmet', exist_ok=True)
    os.makedirs(f'{base_output}/missed_no_helmet', exist_ok=True)
    os.makedirs(f'{base_output}/missed_motorcycle', exist_ok=True)
    
    # Load best model
    model_path = r'C:\Users\title\Downloads\Project_ML\runs\detect\runs\detect\helmet_detection_v16_map=0.88_map95=0.67\weights\best.pt'
    model = YOLO(model_path)
    
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    class_folders = {
        'helmet': f'{base_output}/missed_helmet',
        'motorcycle': f'{base_output}/missed_motorcycle',
        'no_helmet': f'{base_output}/missed_no_helmet'
    }
    
    val_imgs_dir = 'dataset/valid/images'
    val_labels_dir = 'dataset/valid/labels'
    
    false_negatives_by_class = {'helmet': set(), 'motorcycle': set(), 'no_helmet': set()}
    processed = set()
    
    print("🔍 Extracting false negative images...")
    
    for img_file in sorted(os.listdir(val_imgs_dir)):
        if not img_file.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = os.path.join(val_imgs_dir, img_file)
        label_file = os.path.join(val_labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        # Get ground truth
        gt_classes = set(load_annotations(label_file))
        
        # Get predictions
        results = model.predict(img_path, conf=0.3, verbose=False)
        pred_classes = set()
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                pred_classes.add(int(box.cls[0].item()))
        
        # Find missing classes
        missing_classes = gt_classes - pred_classes
        
        for missing_cls in missing_classes:
            cls_name = class_names[missing_cls]
            false_negatives_by_class[cls_name].add(img_file)
            
            # Copy image to folder
            dest_path = os.path.join(class_folders[cls_name], img_file)
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)
                processed.add(img_file)
    
    # Print summary
    print("\n" + "="*70)
    print("FALSE NEGATIVES EXTRACTED")
    print("="*70)
    
    total_fn = sum(len(v) for v in false_negatives_by_class.values())
    
    print(f"\n📁 Output folder: {base_output}/\n")
    
    for cls_name, count in sorted(false_negatives_by_class.items()):
        unique_images = len(count)
        print(f"❌ Missed {cls_name:12} : {unique_images:3d} images")
        print(f"   Folder: {class_folders[cls_name]}/")
        if unique_images > 0:
            print(f"   Examples:")
            for img in list(count)[:3]:
                print(f"      - {img}")
        print()
    
    print("="*70)
    print(f"✓ Total unique images with FN: {len(processed)}")
    print(f"✓ Copied to: {base_output}/")
    print("="*70)

if __name__ == "__main__":
    extract_false_negatives()
