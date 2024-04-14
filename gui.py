

import tkinter as tk
from tkinter import ttk
import cv2
from ytwala import recognitionCode
import numpy as np
from plyer import notification
import datetime
import cv2
import face_recognition
import csv
# from playsound import playsound
import threading
import time

camera_combobox = None

def get_available_cameras():
    """
    Get a list of available cameras.

    Returns:
        list: A list of available cameras in the format "Camera <index>".
    """
    available_cameras = []
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