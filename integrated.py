import numpy as np
from plyer import notification
import datetime
import cv2
import face_recognition
import csv
# from playsound import playsound
import threading
import time


import tkinter as tk
from tkinter import ttk

def recognitionCode():
    
    def notify_me(title, message):
        notification.notify(
            title=title,
            message=message,
            timeout=10
        )

    fire_cascade = cv2.CascadeClassifier("D:/BE PROJECT/face recognization/test/fire_detection.xml")

    known_image1 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/ayush.jpg")
    known_face_encoding1 = face_recognition.face_encodings(known_image1)[0]
    known_image2 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/rakesh.jpg")
    known_face_encoding2 = face_recognition.face_encodings(known_image2)[0]
    known_image3 = face_recognition.load_image_file("D:/BE PROJECT/face recognization/AG.jpg")
    known_face_encoding3 = face_recognition.face_encodings(known_image3)[0]

    known_face_encodings = [known_face_encoding1, known_face_encoding2, known_face_encoding3]
    known_face_names = ["Ayush Naik", "Rakesh Mali", "Atharv Gawande"]

    face_locations = []
    face_encodings = []
    face_names = []

    def face_recognition_task(frame, process_this_frame):
        global face_encodings, face_names
        if process_this_frame:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_names = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    face_names.append(name)
                    notify_me(name + " SPOTTED", "CURRENT TIME: " + curr_time)
                    print(name)

                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Display the name
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom + 30), font, 1.0, (255, 255, 255), 1)

    def fire_detection_task(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fires = fire_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in fires:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            curr_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            notify_me("FIRE SPOTTED", "CURRENT TIME: " + curr_time)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        process_this_frame = True  # Move process_this_frame inside the loop

        thread1 = threading.Thread(target=face_recognition_task, args=(frame, process_this_frame))
        thread2 = threading.Thread(target=fire_detection_task, args=(frame,))

        thread1.start()
        thread2.start()

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()






camera_combobox = None

def get_available_cameras():
    """
    Get a list of available cameras.

    Returns:
        list: A list of available cameras in the format "Camera <index>".
    """
    available_cameras = []
    global camera_obj
    
    for i in range(10):  # Check up to 10 cameras
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            break
        available_cameras.append(f"Camera {i+1}")
        cap.release()
    return available_cameras

def start_camera():
    selected_camera = camera_combobox.get()
    # Perform action with selected camera, e.g., start camera feed
    print(f"Starting camera feed for {selected_camera}")

def login():
    # Create a new window for camera selection
    camera_window = tk.Toplevel(window)
    camera_window.title("Camera Selection")

    # Creating camera selection widgets
    camera_label = ttk.Label(camera_window, text="Select Camera:")
    camera_label.pack(pady=10)
    available_cameras = get_available_cameras()
    camera_combobox = ttk.Combobox(camera_window, values=available_cameras)
    camera_combobox.pack(pady=10)
    submit_button = ttk.Button(camera_window, text="Start Camera", command=recognitionCode)
    submit_button.pack(pady=10)

# Main window
window = tk.Tk()
window.title("Login")

frame = tk.Frame(bg='#333333')

# Creating widgets for login interface
login_label = tk.Label(frame, text="Login", bg='#333333', fg="#FF3399", font=("Arial", 30))
username_label = tk.Label(frame, text="Username", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
username_entry = tk.Entry(frame, font=("Arial", 16))
password_entry = tk.Entry(frame, show="*", font=("Arial", 16))
password_label = tk.Label(frame, text="Password", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
login_button = tk.Button(frame, text="Login", bg="#FF3399", fg="#FFFFFF", font=("Arial", 16), command=login)

# Placing widgets on the screen
login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
username_label.grid(row=1, column=0)
username_entry.grid(row=1, column=1, pady=20)
password_label.grid(row=2, column=0)
password_entry.grid(row=2, column=1, pady=20)
login_button.grid(row=3, column=0, columnspan=2, pady=30)

frame.pack()

window.mainloop()




