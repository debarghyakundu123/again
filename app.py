import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose model
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

def process_frame(frame, pose, counter, status):
    """Process each frame to analyze sit-up movements."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:  # Check if pose_landmarks is not None
            landmarks = results.pose_landmarks.landmark
            sit_up_exercise = SitUpExercise(landmarks)
            counter, status = sit_up_exercise.perform_sit_up(counter, status)

        # Render detections
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(174, 139, 45), thickness=2, circle_radius=2),
        )
    except Exception as e:
        print(f"Error processing frame: {e}")

    return frame, counter, status

def main():
    st.title("Sit-Up Exercise Tracker")
    st.text("This app tracks your sit-up exercise using your webcam feed.")
    
    # Webcam video feed
    run_app = st.checkbox("Start Tracking")
    FRAME_WINDOW = st.image([])
    
    if run_app:
        cap = cv2.VideoCapture(0)  # Use webcam feed
        cap.set(3, 800)  # Set width
        cap.set(4, 480)  # Set height

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            counter = 0
            status = True

            while run_app:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam not found.")
                    break

                frame, counter, status = process_frame(frame, pose, counter, status)
                
                # Display counters and status
                cv2.putText(frame, f"Sit-Ups: {counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Status: {'Up' if status else 'Down'}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Convert frame for Streamlit
                FRAME_WINDOW.image(frame, channels="BGR")

        cap.release()
    else:
        st.write("Check the box above to start tracking.")

if __name__ == "__main__":
    main()
