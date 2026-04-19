"""
Comprehensive analysis of False Positives, False Negatives, and detection errors
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO

def analyze_validation_results():
    """Analyze validation results from v16 model"""
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    v16_model_path = base_path / 'runs/detect/runs/detect/helmet_detection_v16_map=0.88_map95=0.67/weights/best.pt'
    val_images = base_path / 'dataset/valid/images'
    val_labels = base_path / 'dataset/valid/labels'
    
    # Load model
    print("\n" + "="*80)
    print("COMPREHENSIVE DETECTION ERROR ANALYSIS")
    print("="*80)
    
    print("\n[Loading model...]")
    model = YOLO(str(v16_model_path))
    
    # Class mapping
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    
    # Statistics
    stats = {
        'true_positives': defaultdict(int),
        'false_positives': defaultdict(int),
        'false_negatives': defaultdict(int),
        'total_ground_truth': defaultdict(int),
        'total_predictions': defaultdict(int),
        'confused_with': defaultdict(lambda: defaultdict(int)),  # FN confused with other class
    }
    
    # Process validation images
    print("\n[Processing validation images...]")
    image_files = sorted(list(val_images.glob('*.jpg')) + list(val_images.glob('*.png')))
    
    all_fp_data = []  # For false positives
    all_fn_data = []  # For false negatives
    
    for img_file in tqdm(image_files[:50], desc="Analyzing errors"):  # First 50 for analysis
        label_file = val_labels / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        # Read ground truth
        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    gt_boxes.append({
                        'class_id': class_id,
                        'class_name': class_names[class_id],
                        'x_center': x_center,
                        'y_center': y_center,
                        'w': w,
                        'h': h,
                    })
        
        # Run detection
        results = model(str(img_file), conf=0.3, verbose=False)
        img_h, img_w = results[0].orig_shape
        
        pred_boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                pred_boxes.append({
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'conf': conf,
                })
        
        # Match predictions with ground truth
        matched_gt = set()
        matched_pred = set()
        
        # Calculate IoU
        def iou(box1, box2):
            # Convert normalized coords to pixels
            h, w = img_h, img_w
            gt_x1 = int((box1['x_center'] - box1['w']/2) * w)
            gt_y1 = int((box1['y_center'] - box1['h']/2) * h)
            gt_x2 = int((box1['x_center'] + box1['w']/2) * w)
            gt_y2 = int((box1['y_center'] + box1['h']/2) * h)
            
            pred_x1, pred_y1 = int(box2['x1']), int(box2['y1'])
            pred_x2, pred_y2 = int(box2['x2']), int(box2['y2'])
            
            xi1, yi1 = max(gt_x1, pred_x1), max(gt_y1, pred_y1)
            xi2, yi2 = min(gt_x2, pred_x2), min(gt_y2, pred_y2)
            
            if xi2 < xi1 or yi2 < yi1:
                return 0.0
            
            inter_area = (xi2 - xi1) * (yi2 - yi1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            union_area = gt_area + pred_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Match boxes with IoU threshold 0.5
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                box_iou = iou(gt_box, pred_box)
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = j
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                # True positive
                gt_class = gt_boxes[best_gt_idx]['class_name']
                pred_class = pred_box['class_name']
                
                stats['true_positives'][gt_class] += 1
                stats['total_ground_truth'][gt_class] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
            else:
                # False positive
                stats['false_positives'][pred_box['class_name']] += 1
                all_fp_data.append({
                    'image': img_file.name,
                    'predicted_class': pred_box['class_name'],
                    'confidence': pred_box['conf'],
                    'iou_with_best_gt': best_iou if best_gt_idx >= 0 else 0
                })
        
        # Unmatched ground truth = False negatives
        for j, gt_box in enumerate(gt_boxes):
            if j not in matched_gt:
                stats['false_negatives'][gt_box['class_name']] += 1
                stats['total_ground_truth'][gt_box['class_name']] += 1
                
                # Find closest prediction
                best_iou = 0
                closest_pred = None
                for i, pred_box in enumerate(pred_boxes):
                    if i in matched_pred:
                        continue
                    box_iou = iou(gt_box, pred_box)
                    if box_iou > best_iou:
                        best_iou = box_iou
                        closest_pred = pred_box
                
                all_fn_data.append({
                    'image': img_file.name,
                    'missed_class': gt_box['class_name'],
                    'closest_pred': closest_pred['class_name'] if closest_pred else 'None',
                    'closest_iou': best_iou
                })
    
    return stats, all_fp_data, all_fn_data

def print_detailed_metrics(stats, fp_data, fn_data):
    """Print detailed metrics"""
    print("\n" + "="*80)
    print("DETECTION METRICS BY CLASS")
    print("="*80)
    
    class_names = ['helmet', 'motorcycle', 'no_helmet']
    
    print(f"\n{'Class':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}")
    print("-" * 80)
    
    metrics_data = []
    for class_name in class_names:
        tp = stats['true_positives'][class_name]
        fp = stats['false_positives'][class_name]
        fn = stats['false_negatives'][class_name]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:<15} {tp:<6} {fp:<6} {fn:<6} {precision:<12.4f} {recall:<12.4f} {f1:<10.4f}")
        
        metrics_data.append({
            'Class': class_name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    print("="*80)
    
    # Summary
    total_tp = sum(stats['true_positives'].values())
    total_fp = sum(stats['false_positives'].values())
    total_fn = sum(stats['false_negatives'].values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\n🎯 OVERALL METRICS:")
    print(f"  Total True Positives:  {total_tp}")
    print(f"  Total False Positives: {total_fp}")
    print(f"  Total False Negatives: {total_fn}")
    print(f"  Overall Precision: {overall_precision:.4f}")
    print(f"  Overall Recall:    {overall_recall:.4f}")
    print(f"  Overall F1-Score:  {overall_f1:.4f}")
    
    # False positive analysis
    print("\n" + "="*80)
    print("FALSE POSITIVES ANALYSIS")
    print("="*80)
    print(f"\nTotal FP: {len(fp_data)}")
    
    if fp_data:
        fp_df = pd.DataFrame(fp_data)
        print("\n📊 FP by Predicted Class:")
        fp_counts = fp_df['predicted_class'].value_counts()
        for class_name, count in fp_counts.items():
            pct = count / len(fp_data) * 100
            print(f"  {class_name:<15}: {count:>3} ({pct:>5.1f}%)")
        
        print("\n📊 FP Confidence Distribution:")
        print(f"  High confidence (0.9-1.0):  {len(fp_df[fp_df['confidence'] >= 0.9])}")
        print(f"  Medium confidence (0.5-0.9): {len(fp_df[(fp_df['confidence'] >= 0.5) & (fp_df['confidence'] < 0.9)])}")
        print(f"  Low confidence (0.3-0.5):   {len(fp_df[fp_df['confidence'] < 0.5])}")
    
    # False negative analysis
    print("\n" + "="*80)
    print("FALSE NEGATIVES ANALYSIS")
    print("="*80)
    print(f"\nTotal FN: {len(fn_data)}")
    
    if fn_data:
        fn_df = pd.DataFrame(fn_data)
        print("\n📊 FN by Missed Class:")
        fn_counts = fn_df['missed_class'].value_counts()
        for class_name, count in fn_counts.items():
            pct = count / len(fn_data) * 100
            print(f"  {class_name:<15}: {count:>3} ({pct:>1f}%)")
        
        print("\n📊 What FN was confused with:")
        confusion = fn_df['closest_pred'].value_counts()
        for pred_class, count in confusion.items():
            pct = count / len(fn_data) * 100
            print(f"  {pred_class:<15}: {count:>3} ({pct:>1f}%)")
    
    # Visualization
    print("\n[4/4] Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metrics by class
    df_metrics = pd.DataFrame(metrics_data)
    x_pos = np.arange(len(df_metrics))
    width = 0.25
    
    axes[0, 0].bar(x_pos - width, df_metrics['Precision'], width, label='Precision', alpha=0.8)
    axes[0, 0].bar(x_pos, df_metrics['Recall'], width, label='Recall', alpha=0.8)
    axes[0, 0].bar(x_pos + width, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Precision, Recall, F1-Score by Class')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(df_metrics['Class'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # TP, FP, FN by class
    x_pos = np.arange(len(df_metrics))
    width = 0.25
    axes[0, 1].bar(x_pos - width, df_metrics['TP'], width, label='TP', alpha=0.8, color='green')
    axes[0, 1].bar(x_pos, df_metrics['FP'], width, label='FP', alpha=0.8, color='red')
    axes[0, 1].bar(x_pos + width, df_metrics['FN'], width, label='FN', alpha=0.8, color='orange')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('TP, FP, FN by Class')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(df_metrics['Class'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # FP confidence distribution
    if fp_data:
        fp_df = pd.DataFrame(fp_data)
        axes[1, 0].hist(fp_df['confidence'], bins=20, color='red', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('False Positives Confidence Distribution')
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    # FN by class
    if fn_data:
        fn_df = pd.DataFrame(fn_data)
        fn_counts = fn_df['missed_class'].value_counts()
        axes[1, 1].barh(fn_counts.index, fn_counts.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('False Negatives by Missed Class')
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    chart_path = Path('c:/Users/title/Downloads/Project_ML/detection_errors_analysis.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"✓ Chart saved: {chart_path}")
    plt.close()

def main():
    print("\n🔬 Starting comprehensive error analysis...")
    stats, fp_data, fn_data = analyze_validation_results()
    print_detailed_metrics(stats, fp_data, fn_data)
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
