import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time

# 1. Configuration and Thresholds
EAR_THRESHOLD = 0.30      # If EAR falls below this, eyes are considered closed
CONSECUTIVE_FRAMES = 30   # Frames to wait before sounding alarm for closed eyes
MAR_THRESHOLD = 0.85      # If MAR goes above this, a yawn is detected
AUDIO_FILE = "fahhhhh.mp3" # Replace with your actual audio file

# 2. Initialize Audio
pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)

# 3. Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices for eyes and mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

def euclidean_distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def calculate_aspect_ratio(landmarks, indices):
    # Calculate vertical distances
    v1 = euclidean_distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = euclidean_distance(landmarks[indices[2]], landmarks[indices[4]])
    # Calculate horizontal distance
    h = euclidean_distance(landmarks[indices[0]], landmarks[indices[3]])
    # Aspect Ratio formula
    return (v1 + v2) / (2.0 * h)

# 4. Main Video Loop
cap = cv2.VideoCapture(0)
frame_counter = 0
alarm_playing = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate EAR for both eyes
            left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Calculate MAR for mouth (Yawning)
            mar = calculate_aspect_ratio(landmarks, MOUTH)

            # 5. Drowsiness Logic & Triggers
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
            else:
                frame_counter = 0 # Reset counter if eyes open
                
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "YAWN DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()

            # Display metrics on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Driver Monitor', frame)

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()