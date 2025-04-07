import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
from pygame import mixer
import time
import threading
import os       


mixer.init()
sound = mixer.Sound('alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def play_short_alarm():
    sound.play()
    time.sleep(0.5)
    sound.stop()

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
SCORE_THRESHOLD = 15
ALARM_COOLDOWN = 2

# Indexes for facial landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    frame_counter = 0
    score = 0
    last_alarm_time = 0
    avg_ear = 0.0  # Default EAR value when no face is detected
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # Always display score, status, and EAR
        status = "Awake" if score <= SCORE_THRESHOLD else "Drowsy"
        cv2.putText(frame, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {score}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Process faces if detected
        if len(faces) > 0:
            for face in faces:
                shape = predictor(gray, face)
                shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                
                left_eye = shape[LEFT_EYE]
                right_eye = shape[RIGHT_EYE]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                
                if avg_ear < EAR_THRESHOLD:
                    frame_counter += 1
                    if frame_counter >= EAR_CONSEC_FRAMES:
                        score += 1
                else:
                    frame_counter = max(0, frame_counter - 1)
                    score = max(0, score - 1)
                
                current_time = time.time()
                if score > SCORE_THRESHOLD and (current_time - last_alarm_time) > ALARM_COOLDOWN:
                    cv2.putText(frame, "DROWSY!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    threading.Thread(target=play_short_alarm, daemon=True).start()
                    last_alarm_time = current_time
        else:
            # Gradually decrease score when no face is detected
            score = max(0, score - 1)
            frame_counter = 0
            avg_ear = 0.0  # Reset EAR when no face is detected
        
        cv2.imshow('Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists('shape_predictor_68_face_landmarks.dat'):
        print("Error: Please download shape_predictor_68_face_landmarks.dat")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit()
    if not os.path.exists('alarm.wav'):
        print("Warning: alarm.wav not found. Please add an audio file.")
    
    detect_drowsiness()