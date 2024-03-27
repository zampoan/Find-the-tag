import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
videoPath = "sampleIRL.mp4"
cap = cv2.VideoCapture(videoPath)

# Train model on custom data, weights stored in runs/detect/trainX
# results = model.train(data='data/data.yaml', epochs=25, imgsz=224,) #  device='mps'

# Inference with custom model
model = YOLO("runs/detect/train3/weights/best.pt")
model.predict(source='data/data.yaml')
# results = model.track(source=videoPath, show=True)
