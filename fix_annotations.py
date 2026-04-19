import os
import numpy as np

label_dir = 'finetundata/train/labels'
backup_dir = 'finetundata/train/labels_backup'

# Create backup
os.makedirs(backup_dir, exist_ok=True)

print("="*70)
print("CLEANING POLYGON ANNOTATIONS TO BBOX FORMAT")
print("="*70)

fixed_count = 0
total_count = 0

# Process all label files
for label_file in os.listdir(label_dir):
    if not label_file.endswith('.txt'):
        continue
    
    input_path = os.path.join(label_dir, label_file)
    backup_path = os.path.join(backup_dir, label_file)
    
    # Backup original
    with open(input_path) as f:
        original_lines = f.readlines()
    with open(backup_path, 'w') as f:
        f.writelines(original_lines)
    
    # Process and fix
    fixed_lines = []
    for line in original_lines:
        total_count += 1
        parts = line.strip().split()
        
        if len(parts) < 5:
            # Invalid line, skip
            continue
        
        cls_id = parts[0]
        
        if len(parts) == 5:
            # Already in bbox format
            fixed_lines.append(line)
        else:
            # Polygon format - convert to bbox
            try:
                coords = [float(x) for x in parts[1:]]
                
                # Extract x,y pairs
                xs = coords[0::2]
                ys = coords[1::2]
                
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    
                    # Convert to center format
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    # Clamp to [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    bbox_line = f"{cls_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}\n"
                    fixed_lines.append(bbox_line)
                    fixed_count += 1
                else:
                    fixed_lines.append(line)
            except ValueError:
                # Can't parse, keep original
                fixed_lines.append(line)
    
    # Write fixed version
    with open(input_path, 'w') as f:
        f.writelines(fixed_lines)

print(f"\n✓ Processed {total_count} total annotations")
print(f"✓ Fixed {fixed_count} polygon annotations to bbox format")
print(f"\n📁 Backup saved to: {backup_dir}")
print(f"\n✅ Labels cleaned and saved!")
