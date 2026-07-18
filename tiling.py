import cv2
import os
import math

# --- CONFIGURATION ---
IMG_DIR = "dataset/images"
LABEL_DIR = "dataset/labels"
OUTPUT_DIR = "dataset/tiled_dataset"
TILE_SIZE = 640
OVERLAP = 200

# Output folders
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)


def get_tiles(img_path, label_path):
    img = cv2.imread(img_path)
    h_img, w_img, _ = img.shape

    # Read global labels
    with open(label_path, 'r') as f:
        lines = f.readlines()

    stride = TILE_SIZE - OVERLAP
    nx = math.ceil((w_img - OVERLAP) / stride)
    ny = math.ceil((h_img - OVERLAP) / stride)

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    count = 0

    for i in range(nx):
        for j in range(ny):
            # Calculate tile boundaries
            x1 = i * stride
            y1 = j * stride
            x2 = min(x1 + TILE_SIZE, w_img)
            y2 = min(y1 + TILE_SIZE, h_img)

            # Adjust if the tile is at the right/bottom edge
            x1 = max(0, x2 - TILE_SIZE)
            y1 = max(0, y2 - TILE_SIZE)

            tile_img = img[y1:y2, x1:x2]
            tile_labels = []

            for line in lines:
                cls, x_c, y_c, w, h = map(float, line.split())

                # Convert from normalized coordinates to absolute pixels (global)
                abs_x = x_c * w_img
                abs_y = y_c * h_img
                abs_w = w * w_img
                abs_h = h * h_img

                # Check if the object center is inside the current tile
                if x1 <= abs_x <= x2 and y1 <= abs_y <= y2:

                    new_x_c = (abs_x - x1) / TILE_SIZE
                    new_y_c = (abs_y - y1) / TILE_SIZE

                    new_w = abs_w / TILE_SIZE
                    new_h = abs_h / TILE_SIZE

                    new_w = min(new_w, 1.0)
                    new_h = min(new_h, 1.0)

                    tile_labels.append(f"{int(cls)} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}")

            # Save the tile only if it contains objects
            if tile_labels:
                tile_fn = f"{base_name}_tile_{i}_{j}"
                cv2.imwrite(os.path.join(OUTPUT_DIR, "images", tile_fn + ".jpg"), tile_img)
                with open(os.path.join(OUTPUT_DIR, "labels", tile_fn + ".txt"), "w") as f_out:
                    f_out.write("\n".join(tile_labels))
                count += 1
    return count


def run_tiling():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"--- Starting Tiling for {len(images)} images ---")

    for idx, img_name in enumerate(images):
        label_name = os.path.splitext(img_name)[0] + ".txt"
        img_path = os.path.join(IMG_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, label_name)

        if os.path.exists(label_path):
            num_tiles = get_tiles(img_path, label_path)
            print(f"[{idx + 1}/{len(images)}] Generated {num_tiles} tiles for {img_name}")
        else:
            print(f"[{idx + 1}/{len(images)}] SKIPPED: {img_name} (missing label)")


if __name__ == "__main__":
    run_tiling()
    print("--- Tiling finished! The dataset is ready for training. ---")