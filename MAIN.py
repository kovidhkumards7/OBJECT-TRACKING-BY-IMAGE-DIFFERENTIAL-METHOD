import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture("D:\\KOVIDH KUMAR D S\\KOVIDH KUMAR D S PYTHON\\PYTHON 2022\\PYTHON PROJECTS\\OPEN CV PROJECTS\\OBJECT TRACKING\\2.mp4")


object_detector = cv2.createBackgroundSubtractorMOG2(history = 100, varThreshold = 40)

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    # print(height, width)
    # roi = frame[340:720, 500: 800]

    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # cv2.imshow("Roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)


    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()