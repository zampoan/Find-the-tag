# opencv-contrib-python installed
# help: https://www.youtube.com/watch?v=T588klKBPNo
import numpy as np
import cv2

GREEN = (0, 255, 0)

OBJECT_TRACKERS = {
    'crst': cv2.legacy.TrackerCSRT_create,
    'moss': cv2.legacy.TrackerMOSSE_create,
    'kcf': cv2.legacy.TrackerKCF_create, 
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'boosting': cv2.legacy.TrackerBoosting_create,
}

# for multi tracking
trackers = cv2.legacy.MultiTracker_create()

videoPath = "sample.mp4"

cap = cv2.VideoCapture(videoPath)

while True:

    _, frame = cap.read()

    success, boxes = trackers.update(frame)

    # When tracking fails, remove the information about the object
    if success == False:
        boundingBox = trackers.getObject()

        print(boundingBox)


    for box in boxes:
        x, y, w, h = [int(c) for c in box]

        # show bounding box
        cv2.rectangle(frame, (x,y), (x+w, y+h), GREEN, 2)
        # print(x, y, w, h)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break
    
    if cv2.waitKey(1) == ord('w'):
        # select the object to track
        roi = cv2.selectROI('Tracking', frame)

        # enable tracking using specific tracker
        tracker = OBJECT_TRACKERS['crst']()
        trackers.add(tracker, frame, roi)

cap.release()
cv2.destroyAllWindows()