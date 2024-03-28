import cv2
import os
from ultralytics import YOLO

def main():
    staffDetectedFrame = []
    staffDetectedCords = []
    
    # Load the YOLOv8 model
    model = YOLO('yolov8m.pt', task='detect')

    # Open the video file
    videoPath = "sample.mp4"
    cap = cv2.VideoCapture(videoPath)

    # getFrames(cap)

    # Check to see if trained already
    if os.path.exists('runs/detect/train'):
        # Inference with custom model
        model = YOLO("runs/detect/train/weights/best.pt")

    else:
        # Can run this line in cli instead of code below
        # yolo task=detect mode=train epochs=100 data=data/data.yaml imgsz=640 model=yolov8m.pt

        # Train model on custom data, weights stored in runs/detect/train 
        model.train(data='data/data.yaml', epochs=100, imgsz=640, device='cpu') #  device='mps' for mac silicon

        model = YOLO("runs/detect/train/weights/best.pt", task='detect')

        # Predicting, saved in /runs/detect/predict3
        # yolo task=detect mode=predict model=runs/detect/train/weights/best.pt show=True conf=0.5 source=sample.mp4

    # Read frames   
    while True:
        numberOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        ret, frame = cap.read()

        if ret:

            # detect and track object
            results = model.track(frame, persist=True)

            # plot results
            frame_ = results[0].plot()

            # Extract bounding box
            boxes = results[0].boxes.xyxy.tolist()

            # visualize
            cv2.imshow('frame', frame_)

            # press 'q' to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            currentFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f'{currentFrame} / {numberOfFrames}')

            # When boxes is not empty, get frame where staff is present
            if boxes:
                # round boxes to nearest .2f
                roundedBoxes = [[round(num,2) for num in sublist] for sublist in boxes]

                staffDetectedFrame.append(currentFrame) # [frame]
                staffDetectedCords.append([(roundedBoxes[0][0], roundedBoxes[0][1]), (roundedBoxes[0][2], roundedBoxes[0][3])]) # [(x1,y1), (x2,y2)]

    cap.release()
    cv2.destroyAllWindows()

    return staffDetectedFrame, staffDetectedCords

if __name__=='__main__':
    staffDetectedFrame, staffDetectedCords = main()
    print(staffDetectedFrame)
