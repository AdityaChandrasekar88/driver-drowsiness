import cv2
import numpy as np
from scipy.spatial import distance as dist
import dlib
from pygame import mixer
import time
import threading
import os       

# Initialize mixer and load alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # horizontal distance between mouth corners
    A = dist.euclidean(mouth[0], mouth[6])
    # vertical distance between upper and lower lips
    B = dist.euclidean(mouth[2], mouth[10])  # top and bottom center points
    C = dist.euclidean(mouth[4], mouth[8])   # mid points
    mar = (B + C) / (2.0 * A)
    return mar

def play_short_alarm():
    sound.play()
    time.sleep(0.5)
    sound.stop()

# Constants
EAR_THRESHOLD = 0.25  # Eye aspect ratio threshold
MAR_THRESHOLD = 0.75  # Mouth aspect ratio threshold for yawn
EAR_CONSEC_FRAMES = 20  # Frames for eye closure detection
YAWN_CONSEC_FRAMES = 15  # Frames for yawn detection
SCORE_THRESHOLD = 15    # Score threshold for alarm
ALARM_COOLDOWN = 2      # Seconds between alarms

# Indexes for facial landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))  # Mouth landmarks

def detect_drowsiness():
    cap = cv2.VideoCapture(0)
    eye_frame_counter = 0
    yawn_frame_counter = 0
    score = 0
    last_alarm_time = 0
    avg_ear = 0.0  # Default EAR value
    avg_mar = 0.0  # Default MAR value
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        # Default status
        eye_status = "Open"
        mouth_status = "Closed"
        overall_status = "Awake"
        
        # Process faces if detected
        if len(faces) > 0:
            for face in faces:
                shape = predictor(gray, face)
                shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
                
                left_eye = shape[LEFT_EYE]
                right_eye = shape[RIGHT_EYE]
                mouth = shape[MOUTH]
                
                # Calculate aspect ratios
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                avg_mar = mouth_aspect_ratio(mouth)
                
                # Draw landmarks
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [mouth], True, (0, 255, 0), 1)
                
                # Eye status detection
                if avg_ear < EAR_THRESHOLD:
                    eye_frame_counter += 1
                    eye_status = "Closed"
                else:
                    eye_frame_counter = max(0, eye_frame_counter - 1)
                    eye_status = "Open"
                
                # Mouth status detection
                if avg_mar > MAR_THRESHOLD:
                    yawn_frame_counter += 1
                    mouth_status = "Yawning"
                else:
                    yawn_frame_counter = max(0, yawn_frame_counter - 1)
                    mouth_status = "Closed"
                
                # Update score based on both eye and mouth status
                if eye_status == "Closed" and eye_frame_counter >= EAR_CONSEC_FRAMES:
                    score += 1
                elif mouth_status == "Yawning" and yawn_frame_counter >= YAWN_CONSEC_FRAMES:
                    score += 1
                else:
                    score = max(0, score - 1)
                
                # Determine overall status
                if score > SCORE_THRESHOLD:
                    overall_status = "Drowsy"
                
                # Trigger alarm if drowsy
                current_time = time.time()
                if overall_status == "Drowsy" and (current_time - last_alarm_time) > ALARM_COOLDOWN:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    threading.Thread(target=play_short_alarm, daemon=True).start()
                    last_alarm_time = current_time
        else:
            # No face detected - gradually decrease score
            score = max(0, score - 1)
            eye_frame_counter = 0
            yawn_frame_counter = 0
            avg_ear = 0.0
            avg_mar = 0.0
        
        # Display information
        status_color = (0, 255, 0) if overall_status == "Awake" else (0, 0, 255)
        
        # Main status and score
        cv2.putText(frame, f"Status: {overall_status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Score: {score}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Eye information
        eye_color = (0, 255, 0) if eye_status == "Open" else (0, 0, 255)
        cv2.putText(frame, f"Eye: {eye_status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        
        # Mouth information
        mouth_color = (0, 255, 0) if mouth_status == "Closed" else (0, 0, 255)
        cv2.putText(frame, f"Mouth: {mouth_status}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        cv2.putText(frame, f"MAR: {avg_mar:.2f}", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        
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
   