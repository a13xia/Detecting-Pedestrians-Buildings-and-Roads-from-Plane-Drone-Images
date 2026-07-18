from ultralytics import YOLO
import os

# 1. Load the trained model
model_path = 'runs/train-600-25ep-patience-5/weights/best.pt'
model = YOLO(model_path)

# 2. Define the source for testing
test_source = r'C:\Users\Alexia\IDL\UAVid archive\uavid_test\seq42\Images'

# 3. Run the prediction
model.predict(
    source=test_source,
    conf=0.25,
    save=True,
    imgsz=640,
    project='predictions',
    name='test_predictions-600-25ep-patience-5',
    exist_ok=True
)

print(f"--- Testing Complete ---")
print(f"Results are saved in: predictions/test_predictions-600-25ep-patience-5")