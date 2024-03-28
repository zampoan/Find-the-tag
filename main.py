import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file
videoPath = "sampleIRL.mp4"
cap = cv2.VideoCapture(videoPath)

# Train model on custom data, weights stored in runs/detect/trainX
# results = model.train(data='data/data.yaml', epochs=25, imgsz=224,) #  device='mps'

# Inference with custom model
# model = YOLO("runs/detect/train3/weights/best.pt")
# results = model.track(source=videoPath, show=True)

# Training /runs/detect/train
# yolo task=detect mode=train epochs=100 data=data/data.yaml imgsz=640 model=yolov8m.pt

# Predicting, saved in /runs/detect/predict3
# yolo task=detect mode=predict model=best.pt show=True conf=0.5 source=sample.mp4


# Read frames
# while True:

#     if ret:
        
#         ret, frame = cap.read()

#         # detect and track object
#         results = model.track(frame, persist=True)

#         # plot results
#         frame_ = results[0].plot()

#         # visualize
#         cv2.imshow('frame', frame_)

#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

# cap.release()
# cv2.destroyAllWindows()