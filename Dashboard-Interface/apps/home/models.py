from apps import db

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    session = db.Column(db.String(50), nullable=False)  # Can be session name or timestamp
    date = db.Column(db.String(10), nullable=False)  # Format: YYYY-MM-DD
    status = db.Column(db.String(1), nullable=False)  # '0', '1', or 'P'
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.String(20))  

    student = db.relationship("Student", backref="attendances")

    __table_args__ = (
        db.UniqueConstraint('student_id', 'session', name='unique_student_session'),
        db.CheckConstraint("status IN ('0', '1', 'P')", name='valid_status'),
    )
