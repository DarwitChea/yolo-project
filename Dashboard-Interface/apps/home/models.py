from apps import db

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    session = db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(10), nullable=False)
    present = db.Column(db.Boolean, default=False)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.String(20))

    student = db.relationship("Student")


