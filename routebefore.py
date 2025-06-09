# # -*- encoding: utf-8 -*-
# """
# Copyright (c) 2019 - present AppSeed.us
# """

# from collections import defaultdict
# from datetime import datetime
# import io
# from PIL import Image
# import time
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from apps.home import blueprint
# from flask import jsonify, render_template, request
# from flask_login import login_required
# from jinja2 import TemplateNotFound
# import pandas as pd
# import torch
# from apps.home.models import Attendance


# # Choose device to run detection on: MPS for Mac M1, fallback to CPU
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load model 
# model = YOLO("/Users/zekk/Documents/Code/YoloProject/Yolov11/ClassAttendantWeight_640.pt")
# names = ['Chantra', 'David', 'Kholine', 'Meysorng', 'Monineath', 'Mony',
#          'Nyvath', 'Pheakdey', 'Piseth', 'Sopheak', 'Theary', 'Vatana', 'Vireak']
# model.to(device)

# # Load excel sheet
# excelPath = '/Users/zekk/Documents/Code/YoloProject/Yolov11/attendance.xlsx'

# # Initialize student list & session count
# student_list = [{"Name": name, "Email": f"{name.lower()}@mail.com"} for name in names]
# attendance_df = pd.read_excel(excelPath)
# session_count = 1

# # Initialize dictionary to track detected face, time & date
# detected_faces_set = set()  
# detection_start_times = defaultdict(lambda: None)
# detection_dates = {} 

# # Check if there's a session info column
# def load_session_info():
#     # Look for a specific row/column to store session info
#     session_info_row = attendance_df.iloc[0] 
#     return session_info_row['session_count'], session_info_row['last_session_date']

# @blueprint.route('/start_session', methods=['POST'])
# @login_required
# def start_session():
#     global attendance_df, session_count, excelPath
#     global detected_faces_set, detection_dates, detection_start_times  
    
#     attendance_df = pd.read_excel(excelPath)
#     session_count = sum(col.startswith("Session") for col in attendance_df.columns) + 1

#     today_str = datetime.now().strftime("%d-%m-%Y")
#     session_col = f"Session {session_count}\n{today_str}"

#     if session_col not in attendance_df.columns:
#         attendance_df[session_col] = 0
#         attendance_df.to_excel(excelPath, index=False)
#         session_count += 1

#         # Reset state for new session
#         detected_faces_set.clear()
#         detection_dates.clear()
#         detection_start_times.clear()

#         return jsonify({"message": f"New session '{session_col}' started."}), 200
#     else:
#         return jsonify({"message": f"Session '{session_col}' already exists."}), 200

# @blueprint.route('/process_frame', methods=['POST'])
# @login_required
# def process_frame():
#     global attendance_df, detected_faces_set, detection_dates, detection_start_times

#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     image_file = request.files['image']
#     img = Image.open(io.BytesIO(image_file.read()))
#     img = img.convert('RGB')
#     img_np = np.array(img)
#     resized_frame = cv2.resize(img_np, (640, 480))

#     # Run YOLO inference
#     results = model.predict(source=resized_frame, device=device)[0]

#     detected_faces = []
#     current_time = time.time()
#     today_str = datetime.now().strftime("%d-%m-%Y")

#     # Use the latest session column (assumes last column is session)
#     session_col = attendance_df.columns[-1]

#     for result in results:
#         for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
#             if conf > 0.9:
#                 name = str(model.names[int(cls)])
#                 x1, y1, x2, y2 = box.tolist()
#                 width = x2 - x1
#                 height = y2 - y1

#                 # Track if the person has already been detected in this session
#                 if name not in detected_faces_set:
#                     detected_faces_set.add(name)
#                     detection_dates[name] = today_str
#                     detection_start_times[name] = current_time

#                 # If detected long enough, mark as present
#                 if detection_start_times.get(name):
#                     duration = current_time - detection_start_times[name]
#                     if duration >= 3:
#                         detected_faces.append({
#                             "name": name,
#                             "x": int(x1),
#                             "y": int(y1),
#                             "conf": float(conf),
#                             "width": int(width),
#                             "height": int(height),
#                             "date_detected": detection_dates.get(name)
#                         })

#                         attendance_df.loc[attendance_df["Name"] == name, session_col] = 1

#     attendance_df.to_excel(excelPath, index=False)
#     return jsonify({"detected": detected_faces})


# @blueprint.route('/index')
# @login_required
# def index():
#     global excelPath 

#     df = pd.read_excel(excelPath)
#     columns = df.columns.tolist()
#     data = df.to_dict(orient='records')
    
#     # Pass the DataFrame to the template as a list of dicts
#     # data = attendance_df.to_dict(orient="records")
#     # columns = attendance_df.columns.tolist()  # List of column names (including dynamic session columns)
    
#     return render_template('home/index.html', data=data, columns=columns, segment='index')

# @blueprint.route('/<template>')
# @login_required
# def route_template(template):

#     try:

#         if not template.endswith('.html'):
#             template += '.html'

#         # Detect the current page
#         segment = get_segment(request)

#         # Serve the file (if exists) from app/templates/home/FILE.html
#         return render_template("home/" + template, segment=segment)

#     except TemplateNotFound:
#         return render_template('home/page-404.html'), 404

#     except:
#         return render_template('home/page-500.html'), 500

# def get_segment(request):

#     try:

#         segment = request.path.split('/')[-1]

#         if segment == '':
#             segment = 'index'

#         return segment

#     except:
#         return None
