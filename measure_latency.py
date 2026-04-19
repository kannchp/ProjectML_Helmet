"""
Measure Inference Latency for YOLO Models
YOLOv8n, YOLOv8s, YOLOv10n, YOLOv9s
"""

from ultralytics import YOLO
import time
import glob
import os
import numpy as np

def measure_latency(model_name, num_runs=10):
    """Measure inference latency for a model"""
    print(f"\n⏱️ Measuring latency for {model_name.upper()}...")
    
    try:
        # Load best weights from training
        best_model_path = f'runs/detect/helmet_detection_{model_name}/weights/best.pt'
        model = YOLO(best_model_path)
        print(f"✓ Loaded: {best_model_path}")
    except:
        try:
            # Fallback to pretrained
            model = YOLO(f'{model_name}.pt')
            print(f"✓ Loaded pretrained: {model_name}.pt")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None
    
    # Get validation images
    val_img_dir = 'dataset/valid/images'
    val_images = glob.glob(os.path.join(val_img_dir, '*.jpg')) + \
                 glob.glob(os.path.join(val_img_dir, '*.png'))
    
    if not val_images:
        print(f"❌ No validation images found")
        return None
    
    # Sample first 20 images
    test_images = val_images[:20]
    latencies = []
    
    print(f"📸 Testing on {len(test_images)} images...")
    
    for img_path in test_images:
        # Warm up (skip first inference)
        if len(latencies) == 0:
            model.predict(source=img_path, verbose=False)
        
        # Measure latency
        start = time.time()
        model.predict(source=img_path, verbose=False)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    
    return {
        'model': model_name,
        'avg_latency_ms': avg_latency,
        'min_latency_ms': min_latency,
        'max_latency_ms': max_latency,
        'std_latency_ms': std_latency,
        'num_samples': len(test_images),
    }

def main():
    print("\n" + "="*80)
    print("INFERENCE LATENCY COMPARISON - v8n, v8s, v10n, v9s")
    print("="*80)
    
    models = ['yolov8n', 'yolov8s', 'yolov10n', 'yolov9s']
    results = []
    
    for model_name in models:
        result = measure_latency(model_name)
        if result:
            results.append(result)
    
    # Display results
    if results:
        print("\n" + "="*80)
        print("⚡ LATENCY RESULTS (ms)")
        print("="*80)
        
        print(f"\n{'Model':<12} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Std (ms)':<12}")
        print("-" * 60)
        
        for r in sorted(results, key=lambda x: x['avg_latency_ms']):
            print(f"{r['model']:<12} {r['avg_latency_ms']:<12.2f} {r['min_latency_ms']:<12.2f} {r['max_latency_ms']:<12.2f} {r['std_latency_ms']:<12.2f}")
        
        # Find fastest
        fastest = min(results, key=lambda x: x['avg_latency_ms'])
        print("\n" + "="*80)
        print(f"🏆 FASTEST MODEL: {fastest['model'].upper()}")
        print(f"   Average Latency: {fastest['avg_latency_ms']:.2f} ms")
        print(f"   Min Latency: {fastest['min_latency_ms']:.2f} ms")
        print(f"   Max Latency: {fastest['max_latency_ms']:.2f} ms")
        print("="*80)

if __name__ == '__main__':
    main()
