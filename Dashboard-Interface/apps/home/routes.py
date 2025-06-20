# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from collections import defaultdict
from datetime import datetime
import io
import time
import cv2
import numpy as np
from ultralytics import YOLO
from apps.home import blueprint
from flask import jsonify, render_template, request
from flask_login import login_required
from flask import send_file
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
from jinja2 import TemplateNotFound
import pandas as pd
import torch
import re
from apps import db
from apps.home.models import Attendance
from apps.home.models import Student


# Choose device to run detection on: MPS for Mac M1, fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load model 
model = YOLO("apps/static/models/ClassAttendantWeight_320.pt")
names = ['Chantra', 'David', 'Kholine', 'Meysorng', 'Monineath', 'Mony',
         'Nyvath', 'Pheakdey', 'Piseth', 'Sopheak', 'Theary', 'Vatana', 'Vireak']
model.to(device)

# # Load excel sheet
# excelPath = '/Users/zekk/Documents/Code/YoloProject/Yolov11/attendance.xlsx'

# # Initialize student list & session count
# student_list = [{"Name": name, "Email": f"{name.lower()}@mail.com"} for name in names]
# attendance_df = pd.read_excel(excelPath)
# session_count = 1

# # Initialize dict to track detected face, time & date
# detected_faces_set = set()  
# detection_start_times = defaultdict(lambda: None)
# detection_dates = {} 

# # Check if there's a session info column
# def load_session_info():
#     # Look for a specific row/column to store session info
#     session_info_row = attendance_df.iloc[0] 
#     return session_info_row['session_count'], session_info_row['last_session_date']

# def extract_session_number(session_label):
#     match = re.search(r'Session (\d+)', session_label)
#     return int(match.group(1)) if match else float('inf')

@blueprint.route('/start_session', methods=['POST'])
@login_required
def start_session():
    global detected_faces_set, detection_dates, detection_start_times, session_count

    # Generate session label
    today_str = datetime.now().strftime("%d-%m-%Y")
    existing_sessions = db.session.query(Attendance.session).distinct().all()
    session_count = len(existing_sessions) + 1
    session_col = f"Session {session_count}\n{today_str}"

    # Reset session state
    detected_faces_set.clear()
    detection_dates.clear()
    detection_start_times.clear()

    return jsonify({"message": f"New session '{session_col}' started."}), 200

@blueprint.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    global detected_faces_set, detection_dates, detection_start_times, session_count

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    resized_frame = cv2.resize(img_np, (640, 480))

    results = model.predict(source=resized_frame, device=device)[0]
    detected_faces = []
    confirmed_names = []

    current_time = time.time()
    today_str = datetime.now().strftime("%d-%m-%Y")
    session_col = f"Session {session_count}\n{today_str}"

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf > 0.9:
                name = str(model.names[int(cls)])
                x1, y1, x2, y2 = box.tolist()
                width = x2 - x1
                height = y2 - y1

                detected_faces.append({
                    "name": name,
                    "x": int(x1),
                    "y": int(y1),
                    "conf": float(conf),
                    "width": int(width),
                    "height": int(height),
                    "date_detected": today_str
                })

                # Start timing detection if first seen
                if name not in detection_start_times:
                    detection_start_times[name] = current_time
                    detection_dates[name] = today_str

                duration = current_time - detection_start_times[name]

                # Get student from DB
                student = Student.query.filter_by(name=name).first()
                if not student:
                    continue  # skip unknown faces

                # Avoid duplicate logging
                already_logged = Attendance.query.filter_by(
                    student_id=student.id,
                    session=session_col,
                    date=today_str
                ).first()

                # Only log if held for at least 2s and not already recorded
                if duration >= 2 and not already_logged:
                    record = Attendance(
                        student_id=student.id,
                        session=session_col,
                        date=today_str,
                        status='1',  
                        confidence=float(conf),
                        timestamp=datetime.now().strftime("%H:%M:%S")
                    )
                    db.session.add(record)
                    db.session.commit()
                    confirmed_names.append(name)

    return jsonify({
        "detected": detected_faces,
        "confirmed": confirmed_names
    })
    
@blueprint.route('/export_attendance')
@login_required
def export_attendance():
    import pandas as pd
    from io import BytesIO
    from flask import send_file
    import re

    students = Student.query.all()
    records = Attendance.query.all()

    # Get sessions sorted by session number (e.g., "Session 1\n09-06-2025")
    session_columns = sorted(
        {r.session for r in records},
        key=lambda s: int(re.search(r'\d+', s).group()) if re.search(r'\d+', s) else 0
    )

    # Build attendance map: {student_id: {session: status}}
    attendance_map = {}
    for r in records:
        if r.student_id not in attendance_map:
            attendance_map[r.student_id] = {}
        attendance_map[r.student_id][r.session] = r.status  # '0', '1', or 'P'

    # Build export rows
    data = []
    for idx, student in enumerate(students, start=1):
        row = {
            "No": idx,
            "Name": student.name,
            "Email": student.email
        }
        for session in session_columns:
            row[session] = attendance_map.get(student.id, {}).get(session, '0')  # Default to '0'
        data.append(row)

    df = pd.DataFrame(data)

    # Export to Excel in memory
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output,
        download_name="attendance_report.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
@blueprint.route('/update_attendance', methods=['POST'])
@login_required
def update_attendance():
    try:
        data = request.get_json()
        updates = data.get('updates', [])

        for update in updates:
            student_id = update['student_id']
            session_name = update['session_name']
            status = update['status']

            # Check for existing attendance record
            record = Attendance.query.filter_by(
                student_id=student_id,
                session=session_name
            ).first()

            if record:
                record.status = status
            else:
                new_record = Attendance(
                    student_id=student_id,
                    session=session_name,
                    date=datetime.now().strftime('%d-%m-%Y'),
                    status=status
                )
                db.session.add(new_record)

        db.session.commit()
        return jsonify({'message': 'Attendance updated successfully'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
  
@blueprint.route('/delete_session', methods=['POST'])
@login_required
def delete_session():
    from flask import request, jsonify
    data = request.get_json()
    session_name = data.get('session')

    print(f"Deleting full session key: '{session_name}'")

    if not session_name:
        return jsonify(success=False, message="Session name missing")

    try:
        rows_deleted = Attendance.query.filter_by(session=session_name).delete()
        db.session.commit()

        if rows_deleted == 0:
            return jsonify(success=False, message="No matching session found.")
        return jsonify(success=True)
    except Exception as e:
        db.session.rollback()
        return jsonify(success=False, message=f"SQL error: {str(e)}")


@blueprint.route('/index')
@login_required
def index():
    students = Student.query.all()
    records = Attendance.query.all()

    # 🛠 Sort by extracted session number
    session_columns = sorted(
        {r.session for r in records},
        key=extract_session_number
    )

    # Build attendance map
    attendance_map = {}
    for r in records:
        if r.student_id not in attendance_map:
            attendance_map[r.student_id] = {}
        attendance_map[r.student_id][r.session] = r.status 

    # Build data table
    data = []
    for student in students:
        row = {
            "student_id": student.id,
            "Name": student.name,
            "Email": student.email
        }
        for session in session_columns:
            row[session] = attendance_map.get(student.id, {}).get(session, 0)
        data.append(row)

    columns = ["Name", "Email"] + session_columns
    return render_template('home/index.html', data=data, columns=columns, segment='index')

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500

def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
