import os
import shutil
import random

# --- CONFIGURATION ---
IMAGE_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
FINAL_DIR = "dataset_FULL"
SPLIT_RATIO = 0.8

# Create YOLO directory structure for training and validation
for folder in ['train', 'val']:
    os.makedirs(os.path.join(FINAL_DIR, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(FINAL_DIR, folder, 'labels'), exist_ok=True)

# Get the list of all images (excluding extensions)
images = [os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))]

# Shuffle images for a random distribution
random.shuffle(images)

# Calculate split index based on the ratio
split_idx = int(len(images) * SPLIT_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]


def move_files(img_list, subset):
    """Copies images and their corresponding labels to the target subset folder."""
    print(f"Moving {len(img_list)} files to {subset}...")
    for name in img_list:
        # Check for original image extension (.jpg or .png)
        ext = ".jpg" if os.path.exists(os.path.join(IMAGE_DIR, name + ".jpg")) else ".png"

        # Copy the image file
        shutil.copy(os.path.join(IMAGE_DIR, name + ext),
                    os.path.join(FINAL_DIR, subset, "images", name + ext))

        # Copy the corresponding label file (.txt)
        shutil.copy(os.path.join(LABEL_DIR, name + ".txt"),
                    os.path.join(FINAL_DIR, subset, "labels", name + ".txt"))


# Execute the splitting process
move_files(train_imgs, "train")
move_files(val_imgs, "val")

print(f"Success! 4K dataset is ready in: {FINAL_DIR}")