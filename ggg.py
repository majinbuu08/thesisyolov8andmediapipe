# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import mediapipe as mp
import numpy as np
import requests

# Initialize YOLO model
model = YOLO("C:\\Users\\jdich\\thesisnew\\runs\\detect\\train2\\weights\\best.pt")

classNames = ['person']

# Initialize Mediapipe pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Initialize variables and parameters
flag = 0
counter = 0
fall = 0
sideway_slight = 0
sideway_whole = 0
front = 0
dir1 = {}
body_angle = ""  # Define body_angle variable here

# Main loop to process each frame from the camera
cap = cv2.VideoCapture(0)
cap.set(3, 1024)
cap.set(4, 600)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print('Ignoring empty camera frame')
            break

        # Perform object detection using YOLO
        results = model(img, stream=True)
        detected_boxes = {}  # Dictionary to store the highest confidence box for each class

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] == 'person':
                    conf = box.conf[0]
                    if conf > detected_boxes.get(cls, 0):
                        detected_boxes[cls] = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        cvzone.cornerRect(img, (x1, y1, w, h))
                        cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)),
                                           scale=0.9, thickness=2)

        # Perform pose estimation using Mediapipe
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        lst = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                lst.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

        # Calculate body angle and fall detection parameters
        if results.pose_landmarks:
            shoulder_wide = abs(lst[11][0] - lst[12][0])
            s_h_high = abs((lst[23][1] + lst[24][1] - lst[11][1] - lst[12][1]) / 2)
            s_h_long = np.sqrt(((lst[23][1] + lst[24][1] - lst[11][1] - lst[12][1]) / 2) ** 2 +
                                ((lst[23][0] + lst[24][0] - lst[11][0] - lst[12][0]) / 2) ** 2)
            h_f_high = ((lst[28][1] + lst[27][1] - lst[24][1] - lst[23][1]) / 2)
            h_f_long = np.sqrt(((lst[28][1] + lst[27][1] - lst[24][1] - lst[23][1]) / 2) ** 2 +
                               ((lst[28][0] + lst[27][0] - lst[24][0] - lst[23][0]) / 2) ** 2)
            rate1 = shoulder_wide / s_h_high

            # Update body angle
            if 0.2 < rate1 < 0.4:
                sideway_slight += 1
                sideway_whole = 0
                front = 0
            elif rate1 < 0.2:
                sideway_whole += 1
                sideway_slight = 0
                front = 0
            else:
                sideway_whole = 0
                sideway_slight = 0
                front = 0

            if sideway_slight >= 3:
                sideway_slight = 0
                body_angle = 'sideway slight'
            elif sideway_whole >= 3:
                sideway_whole = 0
                body_angle = 'sideway whole'
            else:
                front += 1
            if front >= 3:
                body_angle = 'front'

            # Define fall detection algorithm parameters
            para_s_h_1 = 1.15
            para_s_h_2 = 0.85
            para_h_f = 0.6
            para_fall_time = 5

            # Perform fall detection
            if s_h_high < s_h_long * para_s_h_1 and s_h_high > s_h_long * para_s_h_2:
                fall = 0
            elif h_f_high < para_h_f * h_f_long:
                fall += 1
            else:
                fall = 0
                print(f'Bend Over')

            if fall >= para_fall_time:
                counter += 1
                fall = 0
                print(lst[0][1], '\t', lst[11][1], '\t', lst[23][1])
                print("Fall detected!")

        else:
            print('No pose landmarks detected.')

        # Update frame with overlays
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display fall frames number and body angle
        cv2.rectangle(img, (0, 0), (225, 130), (245, 117, 16), -1)
        cv2.putText(img, 'Fall Frames Number', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'Body Angle', (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(counter), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(body_angle), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Mediapipe Feed', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
