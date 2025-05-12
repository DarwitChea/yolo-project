import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

# Load trained model
model = YOLO("/Users/zekk/Documents/Code/YoloProject/Yolov11/ClassAttendantWeight_640.pt")
names = ['Chantra', 'David', 'Kholine', 'Meysorng', 'Monineath', 'Mony',
         'Nyvath', 'Pheakdey', 'Piseth', 'Sopheak', 'Theary', 'Vatana', 'Vireak']

master_file = "/Users/zekk/Documents/Code/YoloProject/Yolov11/student-list.xlsx"
attendance_file = "attendance.xlsx"

# Face Detect Conf & Duration to get Detected
confidence_threshold = 0.9
min_detection_duration = 3  # seconds

# === Load master list ===
if os.path.exists(master_file):
    master_df = pd.read_excel(master_file)
else:
    raise FileNotFoundError("master_list.xlsx not found!")

# === Load or initialize attendance sheet ===
if os.path.exists(attendance_file):
    attendance_df = pd.read_excel(attendance_file)
else:
    attendance_df = master_df.copy()

# === Add new session column ===
today_str = datetime.now().strftime("%d-%m-%Y")
session_cols = [col for col in attendance_df.columns if col.startswith("Session")]
next_session = f"Session {len(session_cols) + 1} {today_str}"
attendance_df[next_session] = 0  # Mark all 0 initially

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Can't access webcam"

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# === Video writer setup ===
video_writer = cv2.VideoWriter(
    'attendance_output.mp4', 
    cv2.VideoWriter_fourcc(*'mp4v'), 
    fps, 
    (width, height)
)

# === Detection state ===
logged_names = set()
detection_start_times = defaultdict(lambda: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Yolo Predition Script
    results = model.predict(source=frame, show=False)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    clss = results.boxes.cls.cpu().tolist()
    confs = results.boxes.conf.cpu().tolist()

    annotator = Annotator(frame, line_width=3, example=names)
    current_time = time.time()

    for box, cls, conf in zip(boxes, clss, confs):
        if conf < confidence_threshold:
            continue

        name = names[int(cls)]
        label = f"{name} {conf:.2f}"
        annotator.box_label(box, label=label)

        if name in logged_names:
            continue

        # Track how long this name has been detected
        if detection_start_times[name] is None:
            detection_start_times[name] = current_time
        else:
            duration = current_time - detection_start_times[name]
            if duration >= min_detection_duration:
                logged_names.add(name)
                print(f"[✔] Logged {name} after {duration:.1f}s")

    # Show and record frame
    frame = annotator.result()
    cv2.imshow("Class Attendance - Face Recognition", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Update & Save attendance 
attendance_df[next_session] = attendance_df["Name"].apply(lambda x: 1 if x in logged_names else 0) 
attendance_df.to_excel(attendance_file, index=False)
print(f"[✓] Attendance updated and saved to {attendance_file}")
