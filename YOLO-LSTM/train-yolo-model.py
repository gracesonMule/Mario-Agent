from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

# Train the model
results = model.train(
    data='synth-data.yaml', 
    cfg='custom_hyp.yaml',
    epochs=100_000, 
    imgsz=640, 
    batch=64,
    device=0 # Uses your primary GPU. Use 'cpu' if no GPU is available.
)

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(
    source='SuperMarioBros-3-2-v0', 
    conf=0.5,   # Confidence threshold: only show boxes it is at least 50% sure about
    save=True,  # Saves copies of the images with the boxes drawn on them
    show=True   # Opens a window to display the results in real-time
)