from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

# Train the model
results = model.train(
    data='data.yaml', 
    epochs=100_000, 
    imgsz=256, 
    batch=16,
    device=0 # Uses your primary GPU. Use 'cpu' if no GPU is available.
)

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(
    source='Mario-Dataset-6/test/images', 
    conf=0.5,   # Confidence threshold: only show boxes it is at least 50% sure about
    save=True,  # Saves copies of the images with the boxes drawn on them
    show=True   # Opens a window to display the results in real-time
)