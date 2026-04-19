"""
Fine-tune existing YOLOv8 helmet detection model
Loads a pre-trained model and continues training with lower learning rate
"""

from ultralytics import YOLO
import torch
import os

def main():
    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("GPU not available, using CPU (training will be slow)")
        device = 'cpu'

    # Select which model to fine-tune
    # Use the best model with mAP=0.88, mAP95=0.67
    model_path = './basemodel/basemodel.pt'
    model_name = 'basemodel.pt'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print(f"Available models: {list(model_paths.keys())}")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Fine-tune with lower learning rate
    print("\n" + "="*70)
    print("FINE-TUNING HELMET DETECTION MODEL (Backbone Frozen)")
    print("="*70)
    
    results = model.train(
        data='finetundata/data.yaml',
        epochs=80,  # Fewer epochs for fine-tuning
        imgsz=640,
        batch=8,
        patience=30,  # Early stopping
        device=device,
        project='runs/detect',
        name=f'{model_name}_finetune_v2',
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
        seed=42,
        freeze=10,  # Freeze first 10 layers (backbone)
        # Lower learning rate for fine-tuning
        lr0 = 0.001,
        lrf = 0.01,
        # Moderate augmentation
        degrees=5 ,
        translate=0.1,
        scale=0.3,
        #fliplr=0.5,
        #mosaic=0.4,
        #mixup=0.05,
    )

    # Validation
    print("\n" + "="*70)
    print("VALIDATION AFTER FINE-TUNING")
    print("="*70)
    metrics = model.val(data='finetundata/data.yaml')
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETED")
    print("="*70)
    print(f"Best model saved to: runs/detect/{model_name}_finetune_v2/weights/best.pt")

if __name__ == "__main__":
    main()
