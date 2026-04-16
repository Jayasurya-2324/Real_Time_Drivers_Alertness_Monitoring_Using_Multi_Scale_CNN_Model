#!/usr/bin/env python3
"""
main_dl.py - Deep-learning enhanced driver monitoring (drop-in for main.py)
Features:
 - Tries to load a lightweight eye-state CNN model (Keras .h5 or TFLite) to classify Open/Closed eyes.
 - Falls back to EAR (from drowsiness_detector.py) if model missing.
 - Optionally uses YOLOv8 (ultralytics) to detect mobile phone; if not available, falls back to hand_near_face heuristic.
 - Keeps non-blocking audio alerts and event logging.
Usage:
    python main_dl.py --source webcam
    python main_dl.py --source sample --video_path sample_videos/video.mp4
Notes:
 - To enable YOLO phone detection: pip install ultralytics and ensure yolov8n.pt is available or ultralytics can download it.
 - To enable CNN eye model: place models/eye_state.h5 (or .tflite) in project root.
"""
import argparse, os, time, threading, json
from collections import deque
from datetime import datetime

import cv2, numpy as np

# Try to import tensorflow if available
_HAS_TF = False
try:
    import tensorflow as tf
    _HAS_TF = True
except Exception:
    _HAS_TF = False

# Try ultralytics YOLO if available
_HAS_UL = False
try:
    from ultralytics import YOLO
    _HAS_UL = True
except Exception:
    _HAS_UL = False

# playsound fallback
try:
    from playsound import playsound
    _HAS_PLAYSOUND = True
except Exception:
    _HAS_PLAYSOUND = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_H5 = os.path.join(BASE_DIR, "models", "eye_state.h5")
MODEL_PATH_TFLITE = os.path.join(BASE_DIR, "models", "eye_state.tflite")
EVENT_LOG_PATH = os.path.join(BASE_DIR, "events_log_dl.json")
DROWSY_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "drowsiness_alert.wav")
DISTRACT_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "distraction_alert.wav")

# Import helper functions
from drowsiness_detector import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
from distraction_detector import hand_near_face

# MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 8
DISTRACT_FRAMES = 6
ALERT_COOLDOWN = 6.0

# Load CNN model if present
cnn_model = None
use_tflite = False
tflite_interpreter = None
if _HAS_TF and os.path.exists(MODEL_PATH_H5):
    try:
        cnn_model = tf.keras.models.load_model(MODEL_PATH_H5)
        print("[info] Loaded Keras eye-state model:", MODEL_PATH_H5)
    except Exception as e:
        print("[warn] Could not load .h5 model:", e)
if os.path.exists(MODEL_PATH_TFLITE):
    try:
        import tensorflow as tf
        tflite_interpreter = tf.lite.Interpreter(MODEL_PATH_TFLITE)
        tflite_interpreter.allocate_tensors()
        use_tflite = True
        print("[info] Loaded TFLite model:", MODEL_PATH_TFLITE)
    except Exception as e:
        print("[warn] Could not load TFLite model:", e)

# Load YOLO if available
yolo_model = None
if _HAS_UL:
    try:
        yolo_model = YOLO("yolov8n.pt")  # will download if not present
        print("[info] YOLOv8 loaded for phone detection")
    except Exception as e:
        print("[warn] YOLO load failed:", e)

def play_sound_nonblocking(path):
    if not _HAS_PLAYSOUND:
        print("[audio] playsound not available, skipping:", path)
        return
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

def log_event(events, kind, note=""):
    ts = datetime.utcnow().isoformat() + "Z"
    events.append({"time": ts, "type": kind, "note": note})
    try:
        with open(EVENT_LOG_PATH, "w") as f:
            json.dump(events, f, indent=2)
    except Exception as e:
        print("Could not write event log:", e)

def predict_eye_state_cnn(gray_eye):
    # expects a grayscale eye image
    img = cv2.resize(gray_eye, (64,64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0,-1))  # shape (1,64,64,1)
    if cnn_model is not None:
        p = cnn_model.predict(img)[0][0]
        # p close to 1 -> open, close to 0 -> closed (depends on training)
        return p
    if use_tflite and tflite_interpreter is not None:
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.set_tensor(input_details[0]['index'], img.astype('float32'))
        tflite_interpreter.invoke()
        p = tflite_interpreter.get_tensor(output_details[0]['index'])[0][0]
        return p
    return None

def detect_phone_yolo(frame):
    # returns True if phone detected in frame
    if yolo_model is None:
        return False
    results = yolo_model(frame)
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            if int(b.cls[0]) == 67:  # COCO class id 67 == cell phone
                return True
    return False

def main(args):
    cap = cv2.VideoCapture(0 if args.source == "webcam" else args.video_path)
    if not cap.isOpened():
        print("Cannot open source")
        return

    events = []
    ear_queue = deque(maxlen=EAR_CONSEC_FRAMES)
    distract_queue = deque(maxlen=DISTRACT_FRAMES)
    last_drowsy_alert = 0
    last_distract_alert = 0

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        is_drowsy = False
        is_distracted = False
        ear_val = None

        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0].landmark
            # compute EAR
            try:
                left_ear = eye_aspect_ratio(face_lms, LEFT_EYE_IDX, w, h)
                right_ear = eye_aspect_ratio(face_lms, RIGHT_EYE_IDX, w, h)
                ear_val = (left_ear + right_ear) / 2.0
            except Exception:
                ear_val = None

            # try cnn prediction for eyes if available
            eye_prob = None
            if ear_val is not None and (cnn_model is not None or use_tflite):
                # crop approximate eye region using landmarks (LEFT_EYE_IDX center)
                # compute bounding box around left eye landmarks
                xs = [int(face_lms[i].x * w) for i in LEFT_EYE_IDX + RIGHT_EYE_IDX]
                ys = [int(face_lms[i].y * h) for i in LEFT_EYE_IDX + RIGHT_EYE_IDX]
                x1, x2 = max(min(xs)-5,0), min(max(xs)+5,w)
                y1, y2 = max(min(ys)-5,0), min(max(ys)+5,h)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eye_crop = gray[y1:y2, x1:x2]
                if eye_crop.size != 0:
                    p = predict_eye_state_cnn(eye_crop)
                    if p is not None:
                        eye_prob = float(p)

            # Decision: prefer CNN if available otherwise EAR
            if eye_prob is not None:
                # assume model outputs 1=open, 0=closed (if trained that way)
                closed = eye_prob < 0.5
                ear_queue.append(1.0 if not closed else 0.0)
            elif ear_val is not None:
                ear_queue.append(ear_val)
            else:
                ear_queue.clear()

            if len(ear_queue) == ear_queue.maxlen and all(e < EAR_THRESHOLD for e in ear_queue):
                is_drowsy = True

        else:
            ear_queue.clear()

        # Distraction: try YOLO phone detection or hand heuristic
        phone_detected = False
        if _HAS_UL and yolo_model is not None:
            try:
                phone_detected = detect_phone_yolo(frame)
            except Exception:
                phone_detected = False

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0].landmark if face_results.multi_face_landmarks else None
            for hl in hand_results.multi_hand_landmarks:
                if face_lms and hand_near_face(face_lms, hl.landmark, w, h):
                    distract_queue.append(True)
                    break
            else:
                distract_queue.append(False)
        else:
            distract_queue.append(False)

        # if phone detected and hand near face (or repeated hand), count as distracted
        if phone_detected and any(distract_queue):
            is_distracted = True
        elif len(distract_queue) == distract_queue.maxlen and all(distract_queue):
            is_distracted = True

        now = time.time()
        if is_drowsy and now - last_drowsy_alert > ALERT_COOLDOWN:
            last_drowsy_alert = now
            play_sound_nonblocking(DROWSY_ALERT_PATH)
            log_event(events, "drowsiness", f"ear_queue_mean={np.mean(list(ear_queue)):.3f}")

        if is_distracted and now - last_distract_alert > ALERT_COOLDOWN:
            last_distract_alert = now
            play_sound_nonblocking(DISTRACT_ALERT_PATH)
            log_event(events, "distraction", f"phone_detected={phone_detected}")

        # overlay
        status = "OK"
        color = (0,255,0)
        if is_drowsy:
            status = "DROWSY"; color=(0,0,255)
        elif is_distracted:
            status = "DISTRACTED"; color=(0,165,255)
        cv2.putText(frame, f"Status: {status}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if ear_val is not None:
            cv2.putText(frame, f"EAR: {ear_val:.3f}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Driver Monitor - DL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    try:
        with open(EVENT_LOG_PATH, "w") as f:
            json.dump(events, f, indent=2)
        print("Event log saved to", EVENT_LOG_PATH)
    except Exception as e:
        print("Saving events failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("webcam","sample"), default="webcam")
    parser.add_argument("--video_path", default="sample_videos/video.mp4")
    args = parser.parse_args()
    main(args)
