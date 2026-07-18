from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='uavid_data.yaml',
    epochs=25,
    imgsz=640,
    batch=8,
    patience=5,
    project='runs',
    name='train-600-25ep-patience-5',
    device='cpu'
)