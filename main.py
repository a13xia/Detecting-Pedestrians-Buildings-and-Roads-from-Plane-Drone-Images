import cv2
import numpy as np
import os

# --- CONFIGURATION ---
COLOR_IMAGES_PATH = "dataset/images"
MASKS_PATH = "dataset/masks"
OUTPUT_LABELS_PATH = "dataset/labels"
VISUALIZATION_PATH = "dataset/debug_view"

CLASS_CONFIG = {
    0: {"name": "building", "mask_rgb": [128, 0, 0], "draw_bgr": [0, 0, 255], "min_area": 100},
    1: {"name": "road", "mask_rgb": [128, 64, 128], "draw_bgr": [128, 64, 128], "min_area": 500},
    2: {"name": "pedestrian", "mask_rgb": [64, 64, 0], "draw_bgr": [0, 64, 64], "min_area": 5}
}

os.makedirs(OUTPUT_LABELS_PATH, exist_ok=True)
os.makedirs(VISUALIZATION_PATH, exist_ok=True)


def process_dataset():
    mask_list = [f for f in os.listdir(MASKS_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(mask_list)

    print(f"--- Starting processing of {total} files ---")

    for i, file_name in enumerate(mask_list):
        mask = cv2.imread(os.path.join(MASKS_PATH, file_name))
        img_color = cv2.imread(os.path.join(COLOR_IMAGES_PATH, file_name))

        if mask is None or img_color is None:
            print(f"[{i + 1}/{total}] SKIPPED: {file_name} (missing files)")
            continue

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        h, w, _ = mask.shape
        yolo_labels = []

        for class_id, info in CLASS_CONFIG.items():
            target = np.array(info["mask_rgb"])
            lower = np.clip(target - 15, 0, 255)
            upper = np.clip(target + 15, 0, 255)

            binary_mask = cv2.inRange(mask_rgb, lower, upper)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < info["min_area"]:
                    continue

                x_px, y_px, w_px, h_px = cv2.boundingRect(cnt)
                x_center = (x_px + w_px / 2.0) / w
                y_center = (y_px + h_px / 2.0) / h
                norm_w = w_px / w
                norm_h = h_px / h

                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                cv2.rectangle(img_color, (x_px, y_px), (x_px + w_px, y_px + h_px), info["draw_bgr"], thickness=5)

        # Save data
        label_name = os.path.splitext(file_name)[0] + ".txt"
        with open(os.path.join(OUTPUT_LABELS_PATH, label_name), "w") as f:
            f.write("\n".join(yolo_labels))
        cv2.imwrite(os.path.join(VISUALIZATION_PATH, file_name), img_color)

        print(f"[{i + 1}/{total}] DONE: {file_name}")


if __name__ == "__main__":
    process_dataset()
    print("--- All files have been processed successfully! ---")