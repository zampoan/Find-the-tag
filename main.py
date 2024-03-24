import numpy as np
import cv2

GREEN = (0, 255, 0)

videoPath = "sample.mp4"

cap = cv2.VideoCapture(videoPath)

# Extracts moving objects
objectDetectionMOG = cv2.createBackgroundSubtractorMOG2(
    history=100, 
    varThreshold=50     # higher value less detection less false positive
    )    

while True:
    ret, frame = cap.read() 
    
    height, width, _ = frame.shape # 720, 960
    # Extract region of interest (roi)
    roi = frame[20: 690, 400: 700] 

    # Object detection
    maskMOG = objectDetectionMOG.apply(roi)

    # get rid of shadows
    _, maskMOG = cv2.threshold(maskMOG, 254, 255, cv2.THRESH_BINARY)

    # find the boundaries of all objects
    contours, _ = cv2.findContours(maskMOG, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # calculate area and remove small elements
        area = cv2.contourArea(cnt)
        
        # find area greater than 200 pixels and draws them
        if area > 200:
            # cv2.drawContours(roi, [cnt], -1, GREEN, 2) # draw all countorIdx

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+h), GREEN, 3)


    cv2.imshow('roi', roi)
    cv2.imshow('frame', frame)
    # cv2.imshow('mog mask', maskMOG)



    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
