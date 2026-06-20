import cv2
import mediapipe as mp
import numpy as np
import pygame
import math
import time

# ==============================================================================
# 1. CONFIGURACIÓN Y UMBRALES (THRESHOLDS)
# ==============================================================================

# EAR_THRESHOLD: Mide la apertura del ojo.
# - Si lo AUMENTAS (ej. 0.35): El sistema se vuelve hipersensible. Detectará que estás
#   dormido incluso si solo tienes los ojos un poco entrecerrados o miras hacia abajo.
# - Si lo DISMINUYES (ej. 0.20): El sistema se vuelve estricto. Tendrás que cerrar
#   los ojos por completo y apretarlos para que la alarma suene.
EAR_THRESHOLD = 0.30

# CONSECUTIVE_FRAMES: Mide el tiempo (en cuadros de video) que los ojos deben estar cerrados.
# - Si lo AUMENTAS (ej. 45): Le das más margen de tiempo al algoritmo. Ignorará por completo
#   los parpadeos largos y requerirá que cierres los ojos por más de 1 segundo para pitar.
# - Si lo DISMINUYES (ej. 10): La alarma será instantánea. Podría pitar por error si 
#   haces un parpadeo lento y natural.
CONSECUTIVE_FRAMES = 30

# MAR_THRESHOLD: Mide la apertura vertical de la boca.
# - Si lo AUMENTAS (ej. 1.00): Requerirá que abras la boca de una manera extremadamente
#   exagerada para registrar el bostezo.
# - Si lo DISMINUYES (ej. 0.50): Provocará falsos positivos. Pitará simplemente porque
#   estás hablando, cantando o sonriendo con la boca abierta.
MAR_THRESHOLD = 0.85

# PITCH_THRESHOLD: Mide la inclinación de la cabeza hacia adelante (cabeceo). Valores negativos.
# - Si lo AUMENTAS acercándolo a cero (ej. -5.0): Detectará la caída con MUCHA anticipación.
#   Apenas bajes el mentón unos milímetros, la alarma saltará.
# - Si lo DISMINUYES alejándolo de cero (ej. -25.0): Requerirá que tu cabeza caiga 
#   profundamente, casi tocando tu pecho con el mentón, para detectar el sueño.
PITCH_THRESHOLD = -10.0

# ROLL_THRESHOLD: Mide la caída lateral de la cabeza (hacia el hombro izquierdo o derecho).
# - Si lo AUMENTAS (ej. 35.0): Exigirá que tu cuello se doble drásticamente hacia tu hombro
#   para considerar que te quedaste dormido.
# - Si lo DISMINUYES (ej. 10.0): La alerta saltará con solo inclinar un poco la cabeza 
#   hacia un lado por curiosidad o para mirar el espejo retrovisor.
ROLL_THRESHOLD = 20.0

# Archivo de audio que se reproducirá en las alertas
AUDIO_FILE = "fahhhhh.mp3"


# ==============================================================================
# 2. INICIALIZAR AUDIO
# ==============================================================================
pygame.mixer.init()
pygame.mixer.music.load(AUDIO_FILE)

# ==============================================================================
# 3. INICIALIZAR MEDIA PIPE FACE MESH (Malla Facial)
# ==============================================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices de los puntos faciales (Landmarks) para ojos y boca
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]

# Puntos 2D/3D seleccionados para estimar la postura de la cabeza
# Punta de la nariz, esquinas de los ojos, esquinas de la boca, mentón
FACE_POSE_POINTS = [1, 33, 263, 61, 291, 199] 

def euclidean_distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def calculate_aspect_ratio(landmarks, indices):
    # Calcula las distancias verticales
    v1 = euclidean_distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = euclidean_distance(landmarks[indices[2]], landmarks[indices[4]])
    # Calcula la distancia horizontal
    h = euclidean_distance(landmarks[indices[0]], landmarks[indices[3]])
    # Fórmula de la proporción (Aspect Ratio)
    return (v1 + v2) / (2.0 * h)

def play_alarm():
    # Verifica si el audio ya está sonando para no reiniciarlo encima
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# ==============================================================================
# 4. BUCLE PRINCIPAL DE VIDEO
# ==============================================================================
cap = cv2.VideoCapture(0)
frame_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    img_h, img_w, img_c = frame.shape
    
    # Convertir la imagen BGR (formato OpenCV) a RGB (formato MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # --- A. ANÁLISIS DE OJOS Y BOCA ---
            left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_aspect_ratio(landmarks, MOUTH)

            # --- B. ESTIMACIÓN DE POSTURA DE LA CABEZA ---
            face_2d = []
            face_3d = []

            # 1. Centrar el modelo 3D tomando la nariz como origen local
            nose_x = landmarks[1].x * img_w
            nose_y = landmarks[1].y * img_h      
                 
            for idx in FACE_POSE_POINTS:
                lm = landmarks[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                # Restamos la posición de la nariz y escalamos la profundidad (z)
                face_3d.append([x - nose_x, y - nose_y, lm.z * img_w])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Matriz simulada de la cámara óptica
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 0, img_w / 2],
                                    [0, focal_length, img_h / 2],
                                    [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Resolver PnP para calcular la rotación en el espacio
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            # Extraer Pitch (Cabeceo) del cálculo de Euler
            pitch = angles[0]

            # Solución trigonométrica para extraer el Roll (Inclinación lateral)
            p_left = landmarks[33]
            p_right = landmarks[263]
            dy = p_right.y - p_left.y
            dx = p_right.x - p_left.x
            roll = math.degrees(math.atan2(dy, dx))

            # --- C. LÓGICA DE FATIGA Y ALARMAS ---
            
            # 1. Verificación de Ojos Cerrados
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "FATIGA: ¡OJOS CERRADOS!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    play_alarm()
            else:
                frame_counter = 0 # Reiniciar contador si abre los ojos
                
            # 2. Verificación de Bostezo
            if mar > MAR_THRESHOLD:
                cv2.putText(frame, "FATIGA: ¡BOSTEZANDO!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alarm()

            # 3. Verificación de Postura (Caída hacia adelante o a los lados)
            if pitch < PITCH_THRESHOLD:
                cv2.putText(frame, "FATIGA: ¡CABEZA CAYENDO!", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alarm()
            elif abs(roll) > ROLL_THRESHOLD:
                cv2.putText(frame, "FATIGA: ¡CABEZA DE LADO!", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                play_alarm()

            # --- D. MOSTRAR MÉTRICAS EN PANTALLA ---
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (450, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Monitor de Conductor', frame)

    # Presionar 'q' para salir del programa
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()