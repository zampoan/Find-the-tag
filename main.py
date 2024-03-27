import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file
videoPath = "sample.mp4"
cap = cv2.VideoCapture(videoPath)

# Run inference
results = model.track('sample.mp4', show=True, tracker='bytetracker.yaml')