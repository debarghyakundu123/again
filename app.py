from flask import Flask, render_template_string, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# HTML template embedded in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sit-Up Tracker</title>
</head>
<body>
    <h1>Sit-Up Exercise Tracker</h1>
    <img src="{{ url_for('video_feed') }}" alt="Video Feed">
</body>
</html>
"""

mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def detection_body_part(landmarks, body_part_name):
    """Detect the coordinates of a specific body part."""
    return [
        landmarks[mp_pose.PoseLandmark[body_part_name].value].x,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].y,
        landmarks[mp_pose.PoseLandmark[body_part_name].value].visibility
    ]

class SitUpExercise:
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def angle_of_the_abdomen(self):
        """Calculate the angle of the abdomen based on landmarks."""
        left_shoulder = detection_body_part(self.landmarks, "LEFT_SHOULDER")
        left_hip = detection_body_part(self.landmarks, "LEFT_HIP")
        left_knee = detection_body_part(self.landmarks, "LEFT_KNEE")
        return calculate_angle(left_shoulder, left_hip, left_knee)

    def perform_sit_up(self, counter, status):
        """Count sit-up repetitions and determine movement status."""
        angle = self.angle_of_the_abdomen()
        if status:
            if angle < 55:  # Sit-up completed
                counter += 1
                status = False
        else:
            if angle > 105:  # Reset position
                status = True

        return counter, status

def score_table(frame, counter, status):
    """Display exercise score details on the frame."""
    cv2.putText(frame, "Activity : Sit-Up", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Counter : " + str(counter), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Status : " + str(status), (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return frame

def generate_frames():
    """Run the Sit-Up Exercise function using webcam feed."""
    cap = cv2.VideoCapture(0)  # webcam

    cap.set(3, 800)  # width
    cap.set(4, 480)  # height

    # Initialize Mediapipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        counter = 0  # Counter for sit-ups
        status = True  # Movement status

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480), interpolation=cv2.INTER_AREA)
            # Recolor frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            # Process frame with Mediapipe
            results = pose.process(frame)
            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:  # Check if pose_landmarks is not None
                    landmarks = results.pose_landmarks.landmark
                    sit_up_exercise = SitUpExercise(landmarks)
                    counter, status = sit_up_exercise.perform_sit_up(counter, status)
            except Exception as e:
                print(f"Error processing frame: {e}")
                pass

            # Render score table
            frame = score_table(frame, counter, status)

            # Render detections (for landmarks)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255),
                                                       thickness=2,
                                                       circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(174, 139, 45),
                                                       thickness=2,
                                                       circle_radius=2),
            )

            # Encode the frame as a byte stream
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Use Flask's response generator to stream video frames
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

@app.route('/')
def index():
    """Render the index HTML template."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Route for video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
