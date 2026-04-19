"""
Validate and compare helmet detection models v16 vs v17
"""
import json
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def validate_model(model_path, data_yaml, project_name):
    """Validate a model and return metrics"""
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        project='validation_results',
        name=project_name,
        device=0,
        imgsz=640,
        batch=16
    )
    return results

def extract_metrics(results):
    """Extract key metrics from validation results"""
    metrics = {
        'mAP@50': results.box.map50 if hasattr(results.box, 'map50') else results.results_dict.get('metrics/mAP50(B)', 0),
        'mAP@50-95': results.box.map if hasattr(results.box, 'map') else results.results_dict.get('metrics/mAP50-95(B)', 0),
        'precision': results.box.mp if hasattr(results.box, 'mp') else results.results_dict.get('metrics/precision(B)', 0),
        'recall': results.box.mr if hasattr(results.box, 'mr') else results.results_dict.get('metrics/recall(B)', 0),
    }
    return metrics

def main():
    base_path = Path('c:/Users/title/Downloads/Project_ML')
    data_yaml = str(base_path / 'dataset/data.yaml')
    
    print("=" * 80)
    print("HELMET DETECTION MODEL COMPARISON: v16 vs v17")
    print("=" * 80)
    
    # Validate v16
    print("\n[1/4] Validating helmet_detection_v16_map=0.88_map95=0.67...")
    v16_model_path = str(base_path / 'runs/detect/runs/detect/helmet_detection_v16_map=0.88_map95=0.67/weights/best.pt')
    try:
        v16_results = validate_model(v16_model_path, data_yaml, 'v16_validation')
        print("✓ v16 validation completed")
    except Exception as e:
        print(f"✗ Error validating v16: {e}")
        v16_results = None
    
    # Validate v17
    print("\n[2/4] Validating helmet_detection_v17...")
    v17_model_path = str(base_path / 'runs/detect/runs/detect/helmet_detection_v17/weights/best.pt')
    try:
        v17_results = validate_model(v17_model_path, data_yaml, 'v17_validation')
        print("✓ v17 validation completed")
    except Exception as e:
        print(f"✗ Error validating v17: {e}")
        v17_results = None
    
    # Extract and compare metrics
    print("\n[3/4] Extracting metrics...")
    if v16_results and v17_results:
        v16_metrics = {
            'mAP@50': v16_results.box.map50,
            'mAP@50-95': v16_results.box.map,
            'precision': v16_results.box.mp,
            'recall': v16_results.box.mr,
        }
        
        v17_metrics = {
            'mAP@50': v17_results.box.map50,
            'mAP@50-95': v17_results.box.map,
            'precision': v17_results.box.mp,
            'recall': v17_results.box.mr,
        }
        
        # Create comparison report
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print(f"{'Metric':<20} {'v16':<15} {'v17':<15} {'Difference':<15} {'Winner':<10}")
        print("-" * 80)
        
        comparison_data = []
        for metric in v16_metrics.keys():
            v16_val = v16_metrics[metric]
            v17_val = v17_metrics[metric]
            diff = v17_val - v16_val
            winner = "v17" if diff > 0 else "v16" if diff < 0 else "Tie"
            
            print(f"{metric:<20} {v16_val:<15.4f} {v17_val:<15.4f} {diff:<15.4f} {winner:<10}")
            comparison_data.append({
                'Metric': metric,
                'v16': v16_val,
                'v17': v17_val,
                'Difference': diff,
                'Winner': winner
            })
        
        print("=" * 80)
        
        # Save comparison to CSV
        df = pd.DataFrame(comparison_data)
        csv_path = base_path / 'model_comparison_v16_vs_v17.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Comparison saved to: {csv_path}")
        
        # Create visualization
        print("\n[4/4] Creating visualization...")
        metrics_names = list(v16_metrics.keys())
        v16_values = [v16_metrics[m] for m in metrics_names]
        v17_values = [v17_metrics[m] for m in metrics_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart comparison
        x = np.arange(len(metrics_names))
        width = 0.35
        ax1.bar(x - width/2, v16_values, width, label='v16', alpha=0.8, color='#1f77b4')
        ax1.bar(x + width/2, v17_values, width, label='v17', alpha=0.8, color='#ff7f0e')
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Difference chart
        differences = [v17_values[i] - v16_values[i] for i in range(len(metrics_names))]
        colors = ['#2ca02c' if d > 0 else '#d62728' for d in differences]
        ax2.bar(metrics_names, differences, color=colors, alpha=0.8)
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylabel('Difference (v17 - v16)', fontsize=12)
        ax2.set_title('Performance Improvement/Regression', fontsize=14, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        chart_path = base_path / 'model_comparison_v16_vs_v17.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved to: {chart_path}")
        plt.close()
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        v17_wins = sum(1 for d in differences if d > 0)
        v16_wins = sum(1 for d in differences if d < 0)
        print(f"v17 wins: {v17_wins}/4")
        print(f"v16 wins: {v16_wins}/4")
        if v17_wins > v16_wins:
            print("\n✓ VERDICT: Model v17 shows BETTER overall performance")
        elif v16_wins > v17_wins:
            print("\n✓ VERDICT: Model v16 shows BETTER overall performance")
        else:
            print("\n✓ VERDICT: Both models are COMPARABLE")
        print("=" * 80)
    else:
        print("✗ Could not complete validation for one or both models")

if __name__ == "__main__":
    main()
