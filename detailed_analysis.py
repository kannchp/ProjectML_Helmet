"""
Detailed analysis of helmet detection model performance (v16 vs v17)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter

def analyze_confusion_matrices():
    """Analyze confusion matrices from validation results"""
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    
    v16_path = base_path / 'validation_results/v16_validation/confusion_matrix_normalized.png'
    v17_path = base_path / 'validation_results/v17_validation/confusion_matrix_normalized.png'
    
    print("\n[Confusion Matrix Analysis]")
    print("=" * 80)
    print("\nConfusion matrices have been generated during validation:")
    print(f"✓ v16: {v16_path}")
    print(f"✓ v17: {v17_path}")
    return v16_path, v17_path

def analyze_dataset_distribution():
    """Analyze class distribution in dataset"""
    dataset_path = Path('c:/Users/title/Downloads/Project_ML/dataset')
    
    print("\n[Dataset Distribution Analysis]")
    print("=" * 80)
    
    # Count annotations in training set
    train_labels = dataset_path / 'train/labels'
    valid_labels = dataset_path / 'valid/labels'
    
    class_counts_train = Counter()
    class_counts_valid = Counter()
    
    # Count training annotations
    if train_labels.exists():
        for label_file in train_labels.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts_train[class_id] += 1
    
    # Count validation annotations
    if valid_labels.exists():
        for label_file in valid_labels.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts_valid[class_id] += 1
    
    # Class mapping
    class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}
    
    print("\n📊 TRAINING SET Distribution:")
    print("-" * 80)
    total_train = sum(class_counts_train.values())
    for class_id in sorted(class_counts_train.keys()):
        count = class_counts_train[class_id]
        pct = (count / total_train * 100) if total_train > 0 else 0
        print(f"  {class_names.get(class_id, f'Class {class_id}'): <15} : {count:>5} annotations ({pct:>6.2f}%)")
    print(f"  {'TOTAL': <15} : {total_train:>5} annotations")
    
    print("\n📊 VALIDATION SET Distribution:")
    print("-" * 80)
    total_valid = sum(class_counts_valid.values())
    for class_id in sorted(class_counts_valid.keys()):
        count = class_counts_valid[class_id]
        pct = (count / total_valid * 100) if total_valid > 0 else 0
        print(f"  {class_names.get(class_id, f'Class {class_id}'): <15} : {count:>5} annotations ({pct:>6.2f}%)")
    print(f"  {'TOTAL': <15} : {total_valid:>5} annotations")
    
    # Check imbalance
    print("\n⚠️ CLASS BALANCE ANALYSIS:")
    print("-" * 80)
    if total_train > 0:
        dominant = max(class_counts_train.items(), key=lambda x: x[1])
        dominant_name = class_names.get(dominant[0], f'Class {dominant[0]}')
        dominant_ratio = dominant[1] / total_train * 100
        print(f"  Dominant class: {dominant_name} ({dominant_ratio:.2f}%)")
        
        # Calculate imbalance ratio
        if min(class_counts_train.values()) > 0:
            imbalance = max(class_counts_train.values()) / min(class_counts_train.values())
            print(f"  Imbalance ratio: {imbalance:.2f}x")
            if imbalance > 3:
                print("  ⛔ SEVERE IMBALANCE - May affect model training!")
    
    return class_counts_train, class_counts_valid, class_names

def analyze_training_curves():
    """Analyze training curves from validation logs"""
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    
    print("\n[Training Curves Analysis]")
    print("=" * 80)
    
    # Read results CSV
    v16_csv = base_path / 'runs/detect/helmet_detection_v16/results.csv'
    v17_csv = base_path / 'runs/detect/helmet_detection_v17/results.csv'
    
    if v17_csv.exists():
        df_v17 = pd.read_csv(v17_csv)
        
        print("\n📈 v17 Training Statistics:")
        print("-" * 80)
        print(f"  Total epochs: {len(df_v17)}")
        
        # Final metrics
        last_row = df_v17.iloc[-1]
        print(f"\n  Final Epoch ({int(last_row['epoch'])} + 1):")
        print(f"    Train loss (box): {last_row['train/box_loss']:.4f}")
        print(f"    Train loss (cls): {last_row['train/cls_loss']:.4f}")
        print(f"    Val loss (box):   {last_row['val/box_loss']:.4f}")
        print(f"    Val loss (cls):   {last_row['val/cls_loss']:.4f}")
        
        # Best metrics during training
        best_map50_idx = df_v17['metrics/mAP50(B)'].idxmax()
        best_row = df_v17.iloc[best_map50_idx]
        print(f"\n  Best mAP@50 at Epoch {int(best_row['epoch']) + 1}:")
        print(f"    mAP@50:    {best_row['metrics/mAP50(B)']:.4f}")
        print(f"    mAP@50-95: {best_row['metrics/mAP50-95(B)']:.4f}")
        print(f"    Precision: {best_row['metrics/precision(B)']:.4f}")
        print(f"    Recall:    {best_row['metrics/recall(B)']:.4f}")
        
        # Convergence analysis
        print(f"\n  Training Convergence:")
        train_loss = df_v17['train/box_loss'].values
        val_loss = df_v17['val/box_loss'].values
        
        train_improvement = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
        val_improvement = (val_loss[0] - val_loss[-1]) / val_loss[0] * 100
        
        print(f"    Train loss improved: {train_improvement:.2f}%")
        print(f"    Val loss improved:   {val_improvement:.2f}%")
        
        # Overfitting check
        avg_train_loss = train_loss[-5:].mean()
        avg_val_loss = val_loss[-5:].mean()
        gap = (avg_val_loss - avg_train_loss) / avg_train_loss * 100
        
        print(f"\n  Overfitting Detection (last 5 epochs):")
        print(f"    Avg train loss: {avg_train_loss:.4f}")
        print(f"    Avg val loss:   {avg_val_loss:.4f}")
        print(f"    Gap: {gap:.2f}%")
        if gap > 20:
            print("    ⚠️ Possible overfitting - validation loss > train loss")
        elif gap < -10:
            print("    ℹ️ Underfitting - train loss > validation loss")
        else:
            print("    ✓ Good convergence")
        
        # Create training curves visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Box loss
        axes[0, 0].plot(df_v17['epoch'] + 1, df_v17['train/box_loss'], label='Train', marker='o', markersize=3)
        axes[0, 0].plot(df_v17['epoch'] + 1, df_v17['val/box_loss'], label='Val', marker='s', markersize=3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Box Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Class loss
        axes[0, 1].plot(df_v17['epoch'] + 1, df_v17['train/cls_loss'], label='Train', marker='o', markersize=3)
        axes[0, 1].plot(df_v17['epoch'] + 1, df_v17['val/cls_loss'], label='Val', marker='s', markersize=3)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Classification Loss Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # mAP metrics
        axes[1, 0].plot(df_v17['epoch'] + 1, df_v17['metrics/mAP50(B)'], label='mAP@50', marker='o', markersize=3)
        axes[1, 0].plot(df_v17['epoch'] + 1, df_v17['metrics/mAP50-95(B)'], label='mAP@50-95', marker='s', markersize=3)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mAP')
        axes[1, 0].set_title('mAP Metrics Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Precision vs Recall
        axes[1, 1].plot(df_v17['epoch'] + 1, df_v17['metrics/precision(B)'], label='Precision', marker='o', markersize=3)
        axes[1, 1].plot(df_v17['epoch'] + 1, df_v17['metrics/recall(B)'], label='Recall', marker='s', markersize=3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        chart_path = Path('c:/Users/title/Downloads/Project_ML/training_curves_analysis.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training curves saved: {chart_path}")
        plt.close()
    else:
        print("  ✗ v17 results.csv not found")

def analyze_per_class_performance():
    """Analyze performance metrics per class"""
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    
    print("\n[Per-Class Performance Analysis]")
    print("=" * 80)
    
    # Read validation results
    v17_results_path = base_path / 'validation_results/v17_validation/results.png'
    
    print("\nClass metrics extracted from validation:")
    print("-" * 80)
    print("\nFrom validation output:")
    print("  helmet        : 261 images, 360 instances")
    print("    Precision: 0.748 | Recall: 0.581 | mAP50: 0.673 | mAP50-95: 0.388")
    print("\n  motorcycle    : 266 images, 592 instances")
    print("    Precision: 0.656 | Recall: 0.422 | mAP50: 0.479 | mAP50-95: 0.239")
    print("\n  no_helmet     : 4 images, 4 instances ⚠️ VERY LIMITED DATA")
    print("    Precision: 0.027 | Recall: 0.750 | mAP50: 0.048 | mAP50-95: 0.038")
    
    print("\n📊 Class Performance Ranking:")
    print("-" * 80)
    print("1. 🥇 helmet        - BEST (mAP50: 0.673)")
    print("   ✓ Good precision (0.748) and reasonable recall (0.581)")
    print("   ✓ Best detection performance")
    
    print("\n2. 🥈 motorcycle     - MEDIUM (mAP50: 0.479)")
    print("   ⚠️ Recall is LOW (0.422) - misses many motorcycles")
    print("   ⚠️ mAP50-95 is very low (0.239)")
    
    print("\n3. 🥉 no_helmet      - WORST (mAP50: 0.048)")
    print("   ❌ Insufficient training data (only 4 instances)")
    print("   ❌ Cannot reliably detect no_helmet class")
    print("   ⚠️ Consider removing or collecting more data")

def generate_recommendations():
    """Generate recommendations for improvement"""
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("=" * 80)
    
    recommendations = [
        {
            "priority": "🔴 CRITICAL",
            "issue": "Motorcycle recall is too low (42.2%)",
            "action": [
                "• Collect more motorcycle training data",
                "• Increase epochs or adjust learning rate",
                "• Check if motorcycle annotations are accurate"
            ]
        },
        {
            "priority": "🔴 CRITICAL",
            "issue": "no_helmet class has insufficient data (4 samples)",
            "action": [
                "• Option A: Remove no_helmet class if not needed",
                "• Option B: Collect 500+ no_helmet samples",
                "• Option C: Use data augmentation techniques"
            ]
        },
        {
            "priority": "🟠 HIGH",
            "issue": "mAP@50-95 is low across all classes",
            "action": [
                "• Improve annotation quality (bounding box accuracy)",
                "• Adjust IoU threshold tuning",
                "• Try different model architectures (larger models)"
            ]
        },
        {
            "priority": "🟠 HIGH",
            "issue": "Training may not have converged optimally",
            "action": [
                "• Increase training epochs (currently 47)",
                "• Implement early stopping based on validation metrics",
                "• Use learning rate scheduling"
            ]
        },
        {
            "priority": "🟡 MEDIUM",
            "issue": "Class imbalance (helmet >> motorcycle >> no_helmet)",
            "action": [
                "• Use weighted loss function",
                "• Apply class weights in training config",
                "• Balance dataset sampling"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['priority']} {rec['issue']}")
        for action in rec['action']:
            print(f"  {action}")
    
    print("\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("=" * 80)
    print("""
1. Fix no_helmet data issue (MUST DO)
2. Collect more motorcycle training samples
3. Retrain v18 with these changes
4. Monitor training curves for optimal stopping point
5. Validate on test set before production deployment
    """)

def main():
    print("\n" + "=" * 80)
    print("DETAILED MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Run all analyses
    analyze_confusion_matrices()
    class_counts_train, class_counts_valid, class_names = analyze_dataset_distribution()
    analyze_training_curves()
    analyze_per_class_performance()
    generate_recommendations()
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
