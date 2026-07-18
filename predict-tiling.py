from ultralytics import YOLO
import os

# 1. Load the trained model
model_path = 'runs/select-train-folder/weights/best.pt'
model = YOLO(model_path)

# 2. Define the source for testing
test_source = r'C:\Users\Alexia\IDL\UAVid archive\uavid_test\seq21\Images'

# 3. Run the prediction (Inference)
results = model.predict(
    source=test_source,
    conf=0.25,           # Confidence threshold: only show detections > 25% certainty
    save=True,           # Save the visual results (images with boxes)
    save_txt=True,       # Save the coordinates in .txt files
    imgsz=640,           # Must match the resolution used during training
    project='predictions',
    name='test_predictions', # The results will be saved here
    exist_ok=True        # Overwrite the folder if it already exists
)

print(f"--- Testing Complete ---")
print(f"Results are saved in: predictions/test_predictions")