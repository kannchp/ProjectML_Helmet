import json
import os

# Paths
coco_path = 'dataset/train/_annotations.coco.json'
yolo_label_dir = 'dataset/train/labels_from_coco'
os.makedirs(yolo_label_dir, exist_ok=True)

with open(coco_path, 'r', encoding='utf-8') as f:
    coco = json.load(f)

# Build image_id to filename mapping
img_id_to_name = {img['id']: img['file_name'] for img in coco['images']}
img_id_to_size = {img['id']: (img['width'], img['height']) for img in coco['images']}

# Build category id to 0-based class id
cat_id_to_class = {cat['id']: cat['id'] - 1 for cat in coco['categories']}
# Collect all annotations per image
img_to_anns = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    if img_id not in img_to_anns:
        img_to_anns[img_id] = []
    img_to_anns[img_id].append(ann)

count = 0
for img_id, file_name in img_id_to_name.items():
    width, height = img_id_to_size[img_id]
    anns = img_to_anns.get(img_id, [])
    label_lines = []
    for ann in anns:
        cat_id = ann['category_id']
        class_id = cat_id_to_class[cat_id]
        # COCO bbox: [x_min, y_min, width, height] (may be string)
        x, y, w, h = [float(v) for v in ann['bbox']]
        # Convert to YOLO: x_center, y_center, w, h (normalized)
        x_center = x + w / 2
        y_center = y + h / 2
        x_center /= width
        y_center /= height
        w /= width
        h /= height
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    # Write label file
    label_file = os.path.splitext(file_name)[0] + '.txt'
    label_path = os.path.join(yolo_label_dir, label_file)
    with open(label_path, 'w', encoding='utf-8') as f:
        f.writelines(label_lines)
    count += 1

print(f"✓ Converted {count} images from COCO to YOLO format.")
print(f"Labels saved to: {yolo_label_dir}")
