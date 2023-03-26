import cv2
import time
import numpy as np
import os
from HandTrackingModule import HandDetector

# Constants
FOLDER_PATH = "header"
WIDTH, HEIGHT = 1280, 720
POINTER_THICKNESS = 20
DETECTION_CONFIDENCE = 0.7

# Load header images
overlay_list = [cv2.imread(f"{FOLDER_PATH}/{im_path}") for im_path in os.listdir(FOLDER_PATH)]
header = overlay_list[0]
draw_color = (0, 0, 255)

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
hand_detector = HandDetector(detectionConfidence=DETECTION_CONFIDENCE)

# Initialize canvas
img_canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
last_x, last_y = 0, 0

# Main loop
prev_time = 0
while True:
    # Capture frame and flip horizontally
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Detect hands and landmarks
    hand_detector.processImage(img)
    hand_detector.drawLandmarks(img)
    landmark_list = hand_detector.getLandmarkPositions(img, draw=False)

    # Handle finger gestures
    if len(landmark_list) > 0:
        fingers_up = hand_detector.fingersUp()
        x1, y1 = landmark_list[hand_detector.tipIds[1]][1:]
        x2, y2 = landmark_list[hand_detector.tipIds[2]][1:]

        # Selection mode
        if fingers_up[1] and fingers_up[2]:
            last_x, last_y = 0, 0
            if y1 < 100:
                if 60 < x1 < 230:
                    header = overlay_list[0]
                    draw_color = (0, 0, 255)
                elif 380 < x1 < 550:
                    header = overlay_list[1]
                    draw_color = (255, 0, 0)
                elif 700 < x1 < 870:
                    header = overlay_list[2]
                    draw_color = (0, 255, 0)
                elif 1030 < x1 < 1250:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
        # Draw mode
        elif fingers_up[1] and not fingers_up[2]:
            cv2.circle(img, (x1, y1), POINTER_THICKNESS, draw_color, cv2.FILLED)
            if last_x == 0 and last_y == 0:
                last_x, last_y = x1, y1
            cv2.line(img_canvas, (last_x, last_y), (x1, y1), draw_color, POINTER_THICKNESS)
            last_x, last_y = x1, y1
        # Clear canvas when all fingers are up
        if all(x >= 1 for x in fingers_up):
            img_canvas = np.zeros(img_canvas.shape, np.uint8)

    # Merge header and canvas images
    img[0:header.shape[0], 0:header.shape[1]] = header
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Display frame and FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
