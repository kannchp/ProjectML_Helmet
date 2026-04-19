import os
from collections import defaultdict

class_names = {0: 'helmet', 1: 'motorcycle', 2: 'no_helmet'}

def count_classes(label_dir):
    class_count = defaultdict(int)
    image_count = 0
    
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            image_count += 1
            path = os.path.join(label_dir, label_file)
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        class_count[cls_id] += 1
    
    return class_count, image_count

print('='*70)
print('DATASET CLASS DISTRIBUTION ANALYSIS')
print('='*70)

# Train set
train_classes, train_images = count_classes('dataset/train/labels')
print(f'\n📊 TRAIN SET ({train_images} images)')
print('-'*70)
total_train = sum(train_classes.values())
for cls_id in [0, 1, 2]:
    count = train_classes[cls_id]
    pct = count*100/total_train if total_train > 0 else 0
    print(f'{class_names[cls_id]:12}: {count:6d} annotations ({pct:6.1f}%)')
print(f'{"TOTAL":12}: {total_train:6d} annotations')

# Validation set
val_classes, val_images = count_classes('dataset/valid/labels')
print(f'\n📊 VALIDATION SET ({val_images} images)')
print('-'*70)
total_val = sum(val_classes.values())
for cls_id in [0, 1, 2]:
    count = val_classes[cls_id]
    pct = count*100/total_val if total_val > 0 else 0
    print(f'{class_names[cls_id]:12}: {count:6d} annotations ({pct:6.1f}%)')
print(f'{"TOTAL":12}: {total_val:6d} annotations')

# Total
print(f'\n📊 TOTAL DATASET')
print('-'*70)
total_all = total_train + total_val
total_images = train_images + val_images
print(f'Images: {total_images}')
for cls_id in [0, 1, 2]:
    count = train_classes[cls_id] + val_classes[cls_id]
    pct = count*100/total_all if total_all > 0 else 0
    print(f'{class_names[cls_id]:12}: {count:6d} annotations ({pct:6.1f}%)')
print(f'{"TOTAL":12}: {total_all:6d} annotations')

# Imbalance analysis
print(f'\n⚠️  CLASS IMBALANCE RATIO')
print('-'*70)
moto_count = train_classes[1] + val_classes[1]
helmet_count = train_classes[0] + val_classes[0]
no_helmet_count = train_classes[2] + val_classes[2]

if moto_count > 0:
    print(f'Motorcycle (base) : 1.00x ({moto_count})')
    print(f'Helmet            : {helmet_count/moto_count:.2f}x ({helmet_count})')
    print(f'No_Helmet         : {no_helmet_count/moto_count:.2f}x ({no_helmet_count})')
    
    if helmet_count < moto_count * 0.3:
        print(f'\n🔴 CRITICAL: Helmet is way underrepresented!')
    elif helmet_count < moto_count * 0.5:
        print(f'\n🟡 WARNING: Helmet is underrepresented - need class weights')
