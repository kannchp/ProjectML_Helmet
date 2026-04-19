"""
Visualize False Positives and False Negatives with bounding boxes
"""
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

def visualize_errors():
    """Visualize FP and FN with bounding boxes"""
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    v16_model_path = base_path / 'runs/detect/runs/detect/helmet_detection_v16_map=0.88_map95=0.67/weights/best.pt'
    val_images = base_path / 'dataset/valid/images'
    val_labels = base_path / 'dataset/valid/labels'
    output_dir = base_path / 'error_visualization'
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'false_positives').mkdir(exist_ok=True)
    (output_dir / 'false_negatives').mkdir(exist_ok=True)
    (output_dir / 'true_positives').mkdir(exist_ok=True)
    
    # Class mapping and colors
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    colors = {
        'tp': (0, 255, 0),      # Green for TP
        'fp': (0, 0, 255),      # Red for FP
        'fn': (0, 165, 255),    # Orange for FN
        'gt': (255, 255, 0)     # Cyan for GT
    }
    
    # Load model
    print("\n" + "="*80)
    print("VISUALIZING FALSE POSITIVES & FALSE NEGATIVES")
    print("="*80)
    print("\n[Loading model...]")
    model = YOLO(str(v16_model_path))
    
    # Process validation images
    image_files = sorted(list(val_images.glob('*.jpg')) + list(val_images.glob('*.png')))
    
    fp_images = []
    fn_images = []
    tp_count = 0
    fp_count = 0
    fn_count = 0
    
    print("\n[Processing validation images...]")
    for img_file in tqdm(image_files[:100], desc="Finding errors"):
        label_file = val_labels / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        # Read image
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Read ground truth
        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                    
                    x1 = int((x_center - box_w/2) * w)
                    y1 = int((y_center - box_h/2) * h)
                    x2 = int((x_center + box_w/2) * w)
                    y2 = int((y_center + box_h/2) * h)
                    
                    gt_boxes.append({
                        'class_id': class_id,
                        'class_name': class_names[class_id],
                        'box': (x1, y1, x2, y2),
                    })
        
        # Run detection
        results = model(str(img_file), conf=0.3, verbose=False)
        
        pred_boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                pred_boxes.append({
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                })
        
        # Calculate IoU
        def iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            xi1 = max(x1_min, x2_min)
            yi1 = max(y1_min, y2_min)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)
            
            if xi2 < xi1 or yi2 < yi1:
                return 0.0
            
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Match boxes
        matched_gt = set()
        matched_pred = set()
        image_has_errors = False
        viz_image = image.copy()
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                box_iou = iou(pred_box['box'], gt_box['box'])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = j
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                # True positive
                x1, y1, x2, y2 = pred_box['box']
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['tp'], 2)
                cv2.putText(viz_image, f"{pred_box['class_name']}", 
                           (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['tp'], 2)
                tp_count += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                # False positive
                x1, y1, x2, y2 = pred_box['box']
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['fp'], 3)
                cv2.putText(viz_image, f"FP: {pred_box['class_name']} ({pred_box['conf']:.2f})", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['fp'], 2)
                fp_count += 1
                image_has_errors = True
        
        # False negatives
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt:
                x1, y1, x2, y2 = gt_box['box']
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['fn'], 3)
                cv2.putText(viz_image, f"MISS: {gt_box['class_name']}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['fn'], 2)
                fn_count += 1
                image_has_errors = True
        
        if image_has_errors:
            # Save error image
            output_path = output_dir / 'errors' / img_file.name
            os.makedirs(output_dir / 'errors', exist_ok=True)
            cv2.imwrite(str(output_path), viz_image)
    
    print(f"\n✓ Found and visualized errors")
    print(f"  Total TP: {tp_count}")
    print(f"  Total FP: {fp_count}")
    print(f"  Total FN: {fn_count}")
    print(f"\nSaved to: {output_dir / 'errors'}")
    
    # Create a summary visualization showing examples
    print("\n[Creating summary visualization...]")
    create_summary_grid(output_dir, val_images, val_labels, model, class_names, colors)

def create_summary_grid(output_dir, val_images, val_labels, model, class_names, colors):
    """Create a grid visualization of error examples"""
    
    image_files = sorted(list(val_images.glob('*.jpg')) + list(val_images.glob('*.png')))
    
    fp_examples = []
    fn_examples = []
    
    for img_file in image_files[:50]:
        label_file = val_labels / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Read GT
        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_w = float(parts[3])
                    box_h = float(parts[4])
                    
                    x1 = int((x_center - box_w/2) * w)
                    y1 = int((y_center - box_h/2) * h)
                    x2 = int((x_center + box_w/2) * w)
                    y2 = int((y_center + box_h/2) * h)
                    
                    gt_boxes.append({
                        'class_id': class_id,
                        'class_name': class_names[class_id],
                        'box': (x1, y1, x2, y2),
                    })
        
        # Run detection
        results = model(str(img_file), conf=0.3, verbose=False)
        
        pred_boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                pred_boxes.append({
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                })
        
        # Check for errors
        def iou(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            xi1 = max(x1_min, x2_min)
            yi1 = max(y1_min, y2_min)
            xi2 = min(x1_max, x2_max)
            yi2 = min(y1_max, y2_max)
            
            if xi2 < xi1 or yi2 < yi1:
                return 0.0
            
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        matched_gt = set()
        matched_pred = set()
        
        has_fp = False
        has_fn = False
        
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                box_iou = iou(pred_box['box'], gt_box['box'])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = j
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                has_fp = True
        
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt:
                has_fn = True
        
        # Draw and save
        viz_image = image.copy()
        
        # Draw predictions
        for i, pred_box in enumerate(pred_boxes):
            x1, y1, x2, y2 = pred_box['box']
            if i in matched_pred:
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['tp'], 2)
            else:
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['fp'], 2)
        
        # Draw GT
        for j, gt_box in enumerate(gt_boxes):
            x1, y1, x2, y2 = gt_box['box']
            if j not in matched_gt:
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), colors['fn'], 2)
        
        if has_fp and len(fp_examples) < 6:
            fp_examples.append((img_file.stem, cv2.resize(viz_image, (300, 300))))
        
        if has_fn and len(fn_examples) < 6:
            fn_examples.append((img_file.stem, cv2.resize(viz_image, (300, 300))))
        
        if len(fp_examples) >= 6 and len(fn_examples) >= 6:
            break
    
    # Create grid images
    if fp_examples:
        print(f"\n[Creating FP visualization grid with {len(fp_examples)} examples...]")
        grid_fp = np.zeros((600, 1800, 3), dtype=np.uint8)
        for idx, (name, img) in enumerate(fp_examples):
            y = (idx // 3) * 300
            x = (idx % 3) * 300
            grid_fp[y:y+300, x:x+300] = img
        
        fp_path = Path('c:/Users/title/Downloads/Project_ML/false_positives_examples.png')
        cv2.imwrite(str(fp_path), grid_fp)
        print(f"✓ Saved: {fp_path}")
    
    if fn_examples:
        print(f"\n[Creating FN visualization grid with {len(fn_examples)} examples...]")
        grid_fn = np.zeros((600, 1800, 3), dtype=np.uint8)
        for idx, (name, img) in enumerate(fn_examples):
            y = (idx // 3) * 300
            x = (idx % 3) * 300
            grid_fn[y:y+300, x:x+300] = img
        
        fn_path = Path('c:/Users/title/Downloads/Project_ML/false_negatives_examples.png')
        cv2.imwrite(str(fn_path), grid_fn)
        print(f"✓ Saved: {fn_path}")

def main():
    visualize_errors()
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print("="*80)

if __name__ == "__main__":
    main()
