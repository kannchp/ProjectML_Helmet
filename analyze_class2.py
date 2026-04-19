import os
import random
from PIL import Image
import cv2

# Check class 2 annotations
label_dir = 'dataset/train/labels'
image_dir = 'dataset/train/images'

class2_files = []

print("="*70)
print("ANALYZING CLASS 2 (no_helmet) ANNOTATIONS")
print("="*70)

# Find all files with class 2
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        path = os.path.join(label_dir, label_file)
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5 and parts[0] == '2':
                    img_file = label_file.replace('.txt', '.jpg')
                    class2_files.append({
                        'image': img_file,
                        'label_file': label_file,
                        'annotation': line.strip()
                    })

print(f"\n✓ Found {len(class2_files)} class 2 annotations")

# Sample some
if class2_files:
    samples = random.sample(class2_files, min(5, len(class2_files)))
    print(f"\nSample class 2 annotations:")
    print("-"*70)
    for s in samples:
        parts = s['annotation'].split()
        x_center, y_center, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        print(f"Image: {s['image']}")
        print(f"  Annotation: {s['annotation']}")
        print(f"  Box size: {w:.4f} x {h:.4f} (normalized)")
        print(f"  Center: ({x_center:.4f}, {y_center:.4f})")
        
        # Check if box is valid
        if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
            print(f"  🔴 ERROR: Center outside [0,1] range!")
        if w < 0 or w > 1 or h < 0 or h > 1:
            print(f"  🔴 ERROR: Width/Height outside [0,1] range!")
        print()

# Check box size stats
print("\nClass 2 Box Size Statistics:")
print("-"*70)
widths = []
heights = []
for item in class2_files:
    parts = item['annotation'].split()
    if len(parts) >= 5:
        widths.append(float(parts[3]))
        heights.append(float(parts[4]))

if widths:
    print(f"Width:  min={min(widths):.4f}, max={max(widths):.4f}, avg={sum(widths)/len(widths):.4f}")
    print(f"Height: min={min(heights):.4f}, max={max(heights):.4f}, avg={sum(heights)/len(heights):.4f}")
    print(f"Total annotations: {len(class2_files)}")
