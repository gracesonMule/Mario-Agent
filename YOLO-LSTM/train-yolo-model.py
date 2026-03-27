from ultralytics import YOLO

model = YOLO('yolo26n-obb.yaml') 

# Train the model
results = model.train(
    data='path/to/your/data.yaml', 
    epochs=100, 
    imgsz=256, 
    batch=16,
    device=0 # Uses your primary GPU. Use 'cpu' if no GPU is available.
)

model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict(
    source='path/to/your/unseen_test_data', 
    conf=0.5,   # Confidence threshold: only show boxes it is at least 50% sure about
    save=True,  # Saves copies of the images with the boxes drawn on them
    show=True   # Opens a window to display the results in real-time
)