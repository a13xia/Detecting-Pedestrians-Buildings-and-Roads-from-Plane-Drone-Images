from ultralytics import YOLO

# 1. Load a pre-trained model (v8n)
model = YOLO('yolov8n.pt')

# 2. Train the model
results = model.train(
    data='uavid_data_tiling.yaml', # Path to the file created in step 2
    epochs=25,
    imgsz=640,              # though an image is 3840x2160
    batch=16,               # How many images to process at once
    patience=5,
    project='runs',
    name='train-name',
    device='cpu'
)