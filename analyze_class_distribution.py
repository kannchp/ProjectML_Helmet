"""
Analyze Class Distribution and Box Sizes
"""

import os
import numpy as np
from pathlib import Path

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
                        'width': width,
                        'height': height,
                        'area': width * height
                    })
    return boxes

def analyze_distribution():
    """Analyze class and box size distribution"""
    
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    class_stats = {
        'helmet': {'count': 0, 'widths': [], 'heights': [], 'areas': []},
        'motorcycle': {'count': 0, 'widths': [], 'heights': [], 'areas': []},
        'no_helmet': {'count': 0, 'widths': [], 'heights': [], 'areas': []}
    }
    
    train_labels_dir = 'dataset/train/labels'
    val_labels_dir = 'dataset/valid/labels'
    
    print("📊 Analyzing dataset distribution...\n")
    
    # Process training data
    for label_file in os.listdir(train_labels_dir):
        if label_file.endswith('.txt'):
            path = os.path.join(train_labels_dir, label_file)
            boxes = load_annotations(path)
            for box in boxes:
                cls_name = class_names[box['class']]
                class_stats[cls_name]['count'] += 1
                class_stats[cls_name]['widths'].append(box['width'])
                class_stats[cls_name]['heights'].append(box['height'])
                class_stats[cls_name]['areas'].append(box['area'])
    
    # Print summary
    print("="*70)
    print("TRAIN SET - CLASS DISTRIBUTION & BOX SIZES")
    print("="*70)
    print(f"\n{'Class':<15} {'Count':<10} {'Avg Width':<12} {'Avg Height':<12} {'Avg Area':<12}")
    print("-"*70)
    
    total_boxes = sum(stats['count'] for stats in class_stats.values())
    
    for cls_name in ['helmet', 'motorcycle', 'no_helmet']:
        stats = class_stats[cls_name]
        count = stats['count']
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        
        if count > 0:
            avg_width = np.mean(stats['widths'])
            avg_height = np.mean(stats['heights'])
            avg_area = np.mean(stats['areas'])
            
            print(f"{cls_name:<15} {count:<10} {avg_width:<12.4f} {avg_height:<12.4f} {avg_area:<12.4f}")
            print(f"  {'%':<13} {pct:<10.1f}%")
            print(f"  {'size range':<13} W:[{min(stats['widths']):.3f}-{max(stats['widths']):.3f}] "
                  f"H:[{min(stats['heights']):.3f}-{max(stats['heights']):.3f}]")
        else:
            print(f"{cls_name:<15} 0 (No data)")
        print()
    
    # Analyze imbalance
    print("="*70)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*70)
    
    helmet_count = class_stats['helmet']['count']
    motorcycle_count = class_stats['motorcycle']['count']
    no_helmet_count = class_stats['no_helmet']['count']
    
    if motorcycle_count > 0:
        helmet_ratio = helmet_count / motorcycle_count
        no_helmet_ratio = no_helmet_count / motorcycle_count
        
        print(f"\nMotorcycle (base) : {motorcycle_count:5d} (1.00x)")
        print(f"Helmet            : {helmet_count:5d} ({helmet_ratio:.2f}x)")
        print(f"No_Helmet         : {no_helmet_count:5d} ({no_helmet_ratio:.2f}x)")
        
        if helmet_count < motorcycle_count * 0.5:
            print("\n⚠️  WARNING: Helmet is UNDERREPRESENTED - Model may struggle!")
        if no_helmet_count < motorcycle_count * 0.3:
            print("⚠️  WARNING: No_Helmet is SEVERELY UNDERREPRESENTED!")
    
    # Box size analysis
    print("\n" + "="*70)
    print("BOX SIZE COMPARISON")
    print("="*70)
    
    if class_stats['motorcycle']['count'] > 0 and class_stats['helmet']['count'] > 0:
        moto_avg_area = np.mean(class_stats['motorcycle']['areas'])
        helmet_avg_area = np.mean(class_stats['helmet']['areas'])
        
        ratio = moto_avg_area / helmet_avg_area if helmet_avg_area > 0 else 0
        
        print(f"\nAverage box area:")
        print(f"  Motorcycle: {moto_avg_area:.4f}")
        print(f"  Helmet:     {helmet_avg_area:.4f}")
        print(f"  Ratio:      {ratio:.2f}x (motorcycle is {ratio:.2f}x larger)")
        
        if ratio > 2:
            print("\n⚠️  Motorcycle boxes are MUCH LARGER!")
            print("    → This is normal (motorcycles are bigger real objects)")
            print("    → But model may over-predict motorcycle size")

if __name__ == "__main__":
    analyze_distribution()
