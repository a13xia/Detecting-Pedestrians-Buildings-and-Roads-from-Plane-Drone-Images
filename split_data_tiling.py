import os
import shutil
import random

# CONFIGURATION
TILED_DIR = "dataset/tiled_dataset"
FINAL_DIR = "dataset_split_tiling"
SPLIT_RATIO = 0.8

# Create YOLO structure
for folder in ['train', 'val']:
    os.makedirs(os.path.join(FINAL_DIR, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DIR, folder, 'labels'), exist_ok=True)

# Get the list of images (without extension)
images = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(TILED_DIR, "images")) if f.endswith('.jpg')]
random.shuffle(images)

split_idx = int(len(images) * SPLIT_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(img_list, subset):
    print(f"Moving {len(img_list)} files to {subset}...")
    for name in img_list:
        # Copy the image
        shutil.copy(os.path.join(TILED_DIR, "images", name + ".jpg"),
                    os.path.join(FINAL_DIR, subset, "images", name + ".jpg"))
        # Copy the label
        shutil.copy(os.path.join(TILED_DIR, "labels", name + ".txt"),
                    os.path.join(FINAL_DIR, subset, "labels", name + ".txt"))

move_files(train_imgs, "train")
move_files(val_imgs, "val")
print("Done! Now you can use 'dataset_split_tiling' for training.")