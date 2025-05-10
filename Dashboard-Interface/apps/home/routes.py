# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import io
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from apps.home import blueprint
from flask import jsonify, render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import pandas as pd

# Load model once at module level
model = YOLO("/Users/zekk/Documents/Code/YoloProject/Yolov11/ClassAttendantWeight_640.pt")
names = ['Chantra', 'David', 'Kholine', 'Meysorng', 'Monineath', 'Mony',
         'Nyvath', 'Pheakdey', 'Piseth', 'Sopheak', 'Theary', 'Vatana', 'Vireak']

@blueprint.route('/process_frame', methods=['POST'])
@login_required
def process_frame():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    img = Image.open(io.BytesIO(image_file.read()))

    # Convert the image to a format that YOLO can work with (if needed)
    img = img.convert('RGB')

    # YOLO Inference (you may need to adjust this based on how you want to handle the image)
    results = model.predict(img)
    
    detected_faces = []
    
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf > 0.9:
                name = str(model.names[int(cls)])

                x1, y1, x2, y2 = box.tolist()
                print(f"Detected {name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                width = x2 - x1
                height = y2 - y1

                detected_faces.append({
                    "name": name,
                    "x": int(x1),
                    "y": int(y1),
                    "conf": float(conf),
                    "width": int(width),
                    "height": int(height)
                })
    
    return jsonify({"detected": detected_faces})

@blueprint.route('/index')
@login_required
def index():
    
    excelPath = '/Users/zekk/Documents/Code/YoloProject/Yolov11/attendance.xlsx'

    df = pd.read_excel(excelPath)
    columns = df.columns.tolist()
    data = df.to_dict(orient='records')
    
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


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
