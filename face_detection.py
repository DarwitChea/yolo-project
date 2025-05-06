from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import pandas as pd
from datetime import datetime
import time
from collections import defaultdict

# Load your trained model
model = YOLO(
    "/Users/zekk/Documents/Code/YoloProject/Yolov11/ClassAttendantWeight_640.pt")
names = ['Chantra', 'David', 'Kholine', 'Meysorng', 'Monineath', 'Mony',
         'Nyvath', 'Pheakdey', 'Piseth', 'Sopheak', 'Theary', 'Vatana', 'Vireak']

# Initialize webcam
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Can't access webcam"

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    'attendance_output.mp4', fourcc, fps, (width, height))

# Logging setup
log_data = []
logged_names = set()
columns = ["Timestamp", "Name", "Confidence", "X1", "Y1", "X2", "Y2"]
confidence_threshold = 0.8
min_detection_duration = 3  # seconds

# Tracking detection duration
detection_start_times = defaultdict(lambda: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLO prediction
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
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_data.append([timestamp, name, round(
                    conf, 3)] + [round(c, 2) for c in box])
                logged_names.add(name)
                print(f"[✔] Logged {name} after {duration:.1f}s")

    # Show and record annotated frame
    frame = annotator.result()
    cv2.imshow("Class Attendance - Face Recognition", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save record to Excel
df = pd.DataFrame(log_data, columns=columns)
df.to_excel("attendance.xlsx", index=False)
print("Attendance saved to attendance.xlsx")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
