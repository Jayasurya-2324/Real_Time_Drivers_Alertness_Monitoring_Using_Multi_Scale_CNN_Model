
"""
main_overlay.py - Enhanced overlay (face wireframe, colored hand/keypoints, bounding boxes)
Creates same behavior as original main.py but with improved visual overlay.
Run: pip install opencv-python mediapipe numpy playsound ultralytics
      python main_overlay.py
"""

# --- protobuf / mediapipe compatibility check added by assistant ---
import sys
try:
    from google.protobuf import symbol_database as _symbol_database
    # Older/newer protobufs may lack GetPrototype; mediapipe expects it.
    if not hasattr(_symbol_database.Default(), "GetPrototype"):
        raise RuntimeError("Incompatible protobuf detected")
except Exception as _e:
    print("\nERROR: Incompatible 'protobuf' package detected (this causes the AttributeError in mediapipe).")
    print("Solution: run the following in your terminal for the Python environment you use to run this program:\n")
    print("  python -m pip install --upgrade pip\n  python -m pip uninstall -y protobuf\n  python -m pip install protobuf==3.20.1\n")
    print("\nAfter installing protobuf==3.20.1, re-run this script. If you use a virtualenv/conda, run the commands there.\n")
    # Exit so user can fix environment before running further (prevents confusing tracebacks)
    sys.exit(1)
# --- end check ---

import os, time, threading, cv2, numpy as np
import mediapipe as mp
from playsound import playsound
import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")
warnings.filterwarnings("ignore", category=UserWarning)

import csv

# import helper funcs from original project if available
try:
    from drowsiness_detector import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
except Exception as e:
    # fallback: simple EAR compute using landmark indices (same indices expected)
    print("Warning: couldn't import drowsiness_detector, using local EAR. Error:", e)
    LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 387, 385, 263, 373, 380]
    def eye_aspect_ratio(landmarks, eye_idx, w, h):
        pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_idx]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if C==0: return 0.0
        return (A+B) / (2.0*C)

try:
    from distraction_detector import hand_near_face
except Exception as e:
    print("Warning: couldn't import distraction_detector.hand_near_face. Error:", e)
    def hand_near_face(face_lms, hand_lms, w, h):
        # naive proximity: compare distance between nose (1) and hand wrist (0)
        try:
            fx = int(face_lms[1].x * w); fy = int(face_lms[1].y * h)
            hx = int(hand_lms[0].x * w); hy = int(hand_lms[0].y * h)
            return np.hypot(fx-hx, fy-hy) < 120
        except:
            return False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DROWSY_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "drowsiness_alert.wav")
DISTRACT_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "distraction_alert.wav")

# MediaPipe initializations
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

def euclidean(a,b): return np.linalg.norm(np.array(a)-np.array(b))

def compute_ear_from_landmarks(landmarks, eye_idx, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_idx]
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return 0.0 if C==0 else (A+B)/(2.0*C)

def bounding_rect_from_landmarks(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)

def play_alert(path):
    # non-blocking playsound
    def _play():
        try:
            playsound(path)
        except:
            pass
    t = threading.Thread(target=_play, daemon=True)
    t.start()

# Video source: try sample video in project, fallback to webcam
video_path = os.path.join(BASE_DIR, "sample_videos", "video.mp4")
VIDEO_SRC = video_path if os.path.exists(video_path) else 0

cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_drowsy_alert = 0
last_distract_alert = 0
DROWSY_COOLDOWN = 5.0
DISTRACT_COOLDOWN = 5.0

window_name = "Driver Monitor - Press Q to quit"


# --- CSV debug/logging setup (assistant added) ---
import sys
csv_path = os.path.join(os.getcwd(), "output_data.csv")
try:
    # Open in append mode so repeated runs don't overwrite during testing.
    first_time = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline='')
    csv_writer = __import__("csv").writer(csv_file)
    if first_time:
        try:
            csv_writer.writerow(["timestamp","frame_index","status","drowsy","distracted","ear","fps"])
            csv_file.flush()
        except Exception as _e:
            print("Warning: failed to write CSV header:", _e, file=sys.stderr)
    print(f"[CSV DEBUG] Logging to: {csv_path}", file=sys.stderr)
except Exception as e:
    print("[CSV DEBUG] Could not open CSV for logging:", e, file=sys.stderr)
    csv_file = None
    csv_writer = None
# --- end debug setup ---


while True:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_face = face_mesh.process(frame_rgb)
    results_hands = hands.process(frame_rgb)

    drowsy = False
    distracted = False
    ear_val = None

    if results_face.multi_face_landmarks:
        face_lms = results_face.multi_face_landmarks[0].landmark
        # draw white wireframe mesh (tesselation + contours)
        mp_drawing.draw_landmarks(frame, results_face.multi_face_landmarks[0],
                                  mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
        mp_drawing.draw_landmarks(frame, results_face.multi_face_landmarks[0],
                                  mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
        # EAR using helper if available
        try:
            ear_left = eye_aspect_ratio(face_lms, LEFT_EYE_IDX, iw, ih)
            ear_right = eye_aspect_ratio(face_lms, RIGHT_EYE_IDX, iw, ih)
        except:
            ear_left = compute_ear_from_landmarks(face_lms, LEFT_EYE_IDX, iw, ih)
            ear_right = compute_ear_from_landmarks(face_lms, RIGHT_EYE_IDX, iw, ih)
        ear_val = (ear_left + ear_right) / 2.0
        if ear_val < 0.22:
            drowsy = True

        # face bounding box
        xs = [int(lm.x * iw) for lm in face_lms]
        ys = [int(lm.y * ih) for lm in face_lms]
        fx1, fy1, fx2, fy2 = max(0, min(xs)-20), max(0, min(ys)-20), min(iw, max(xs)+20), min(ih, max(ys)+20)
        # Hands
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # draw red dots for keypoints
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x*iw), int(lm.y*ih)
                cv2.circle(frame, (x,y), 4, (0,0,255), -1)
            # draw connections in white
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=0))
            # bounding box for hand
            hx1, hy1, hx2, hy2 = bounding_rect_from_landmarks(hand_landmarks.landmark, iw, ih)
            pad=10
            hx1, hy1 = max(0, hx1-pad), max(0, hy1-pad)
            hx2, hy2 = min(iw, hx2+pad), min(ih, hy2+pad)
            color = (255,0,0) if idx==0 else (0,0,255)
            cv2.putText(frame, f"Hand {idx+1}", (hx1, hy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # check hand near face using project's helper if available
            if results_face.multi_face_landmarks:
                if hand_near_face(face_lms, hand_landmarks.landmark, iw, ih):
                    distracted = True

    # Alerts
    if drowsy and time.time()-last_drowsy_alert > DROWSY_COOLDOWN:
        play_alert(DROWSY_ALERT_PATH)
        last_drowsy_alert = time.time()
    if distracted and time.time()-last_distract_alert > DISTRACT_COOLDOWN:
        play_alert(DISTRACT_ALERT_PATH)
        last_distract_alert = time.time()

    # Overlay texts
    if ear_val is not None:
        cv2.putText(frame, f"EAR: {ear_val:.3f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    status = "Normal"
    color = (0,255,0)
    if drowsy and distracted:
        status = "DROWSY + DISTRACTED"
        color = (0,0,255)
    elif drowsy:
        status = "DROWSY"
        color = (0,0,255)
    elif distracted:
        status = "DISTRACTED (Device)"
        color = (0,165,255)
    cv2.putText(frame, status, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 6)

    # FPS
    fps = int(1.0/(time.time()-start+1e-6))
    cv2.putText(frame, f"FPS: {fps}", (iw-140,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200),2)


    # --- CSV write (debug-safe) ---
    try:
        if csv_writer is not None:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            try:
                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            except:
                frame_idx = -1
            status_str = status if 'status' in locals() else ''
            drowsy_flag = int(bool(drowsy)) if 'drowsy' in locals() else 0
            distracted_flag = int(bool(distracted)) if 'distracted' in locals() else 0
            ear_val_safe = round(ear_val,3) if ('ear_val' in locals() and ear_val is not None) else ''
            csv_writer.writerow([ts, frame_idx, status_str, drowsy_flag, distracted_flag, ear_val_safe, fps])
            # flush occasionally to ensure writes appear even if program terminated
            csv_file.flush()
            # debug print every 100 frames
            if frame_idx % 100 == 0:
                print(f"[CSV DEBUG] Wrote frame {frame_idx} to {csv_path}", file=sys.stderr)
    except Exception as e:
        print("[CSV DEBUG] Failed to write CSV row:", e, file=sys.stderr)
    # --- end CSV write ---

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if 'csv_file' in globals():
    try:
        csv_file.close()
    except:
        pass
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
