from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

# Function for speech synthesis
def speak(text):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)

# Load trained data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)[:700]  # Ensure LABELS matches the number of samples in FACES
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Verify the shapes of FACES and LABELS
print('Shape of Faces matrix --> ', FACES.shape)
print('Length of LABELS --> ', len(LABELS))

# Ensure FACES and LABELS have the same number of samples
if FACES.shape[0] != len(LABELS):
    raise ValueError("Number of samples in FACES and LABELS do not match.")

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load and resize background image
imgBackground = cv2.imread("OIP.png")
if imgBackground is not None:
    imgBackground = cv2.resize(imgBackground, (640, 480))

# Constants for CSV file
COL_NAMES = ['NAME', 'TIME']

# Main loop for capturing and processing frames
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and resize detected face
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        # Predict using KNN classifier
        output = knn.predict(resized_img)
        
        # Timestamp for attendance record
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        # Draw rectangles and text on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Prepare attendance data
        attendance = [str(output[0]), str(timestamp)]

    # Overlay frame on background image
    if 'imgBackground' in locals() and imgBackground is not None:
        # Determine the position and size of the frame on the background image
        frame_height, frame_width = frame.shape[:2]
        background_height, background_width = imgBackground.shape[:2]

        x_offset = (background_width - frame_width) // 2
        y_offset = (background_height - frame_height) // 2

        # Place the frame onto the background image
        imgBackground[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = frame
        cv2.imshow("Frame", imgBackground)
    else:
        cv2.imshow("Frame", frame)

    # Keyboard input handling
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        # Write attendance to CSV file
        attendance_file = f"Attendance/Attendance_{date}.csv"
        if os.path.isfile(attendance_file):
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

        # Break out of the while loop to exit
        break

    elif k == ord('q'):
        break

# Release video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
