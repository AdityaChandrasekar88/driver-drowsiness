import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    # horizontal distance
    A = dist.euclidean(mouth[0], mouth[6])
    # vertical distance
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    return (B + C) / (2.0 * A)

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    features = []
    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        
        # Extract eye landmarks
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        
        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Extract mouth landmarks
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)
        
        features.append([avg_ear, mar])
    
    return features[0] if features else None

def process_dataset(dataset_path):
    data = []
    labels = []
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            features = extract_features(image_path)
            
            if features is not None:
                data.append(features)
                labels.append(class_name)
    
    return np.array(data), np.array(labels)

# Path to your dataset
dataset_path = "yawn_eye_dataset_new"
X, y = process_dataset(dataset_path)

# Save the processed data
df = pd.DataFrame(X, columns=['EAR', 'MAR'])
df['label'] = y
df.to_csv('drowsiness_features.csv', index=False)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
print("Train accuracy:", clf.score(X_train, y_train))
print("Test accuracy:", clf.score(X_test, y_test))

# Save the model
joblib.dump(clf, 'drowsiness_model.pkl')