"""
 main_overlay.py
"""
import os, time, threading, cv2, numpy as np
import mediapipe as mp
from playsound import playsound
import warnings, sys, csv
warnings.filterwarnings("ignore")

# 🔥 YOUR PERFECT MODULES
try:
    from drowsiness_detector import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
    print("✅ YOUR drowsiness_detector LOADED")
except:
    print("Using fallback EAR")
    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 387, 385, 263, 373, 380]
    def eye_aspect_ratio(landmarks, eye_idx, w, h):
        pts = np.array([(landmarks[i].x*w, landmarks[i].y*h) for i in eye_idx])
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0

try:
    from distraction_detector import hand_near_face
    print("✅ YOUR distraction_detector LOADED")
except:
    print("Using fallback hand_near_face")
    def hand_near_face(face_lms, hand_lms, w, h):
        try:
            nose = np.array([face_lms[1].x*w, face_lms[1].y*h])
            wrist = np.array([hand_lms[0].x*w, hand_lms[0].y*h])
            return np.linalg.norm(nose - wrist) < 100
        except: return False

#  ULTRA LOW-LIGHT
def preprocess_ultra_lowlight(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    mean_b = np.mean(enhanced[:,:,0])
    gamma = 1.6 if mean_b < 60 else 1.3 if mean_b < 100 else 1.1
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(enhanced, table)
    return cv2.bilateralFilter(gamma_corrected, 9, 75, 75)

#  FIXED MOUTH RATIO (YOUR PROVEN 4 POINTS)
def mouth_aspect_ratio(landmarks, w, h):
    mouth_left = np.array([landmarks[61].x*w, landmarks[61].y*h])
    mouth_right = np.array([landmarks[291].x*w, landmarks[291].y*h])
    mouth_top = np.array([landmarks[13].x*w, landmarks[13].y*h])
    mouth_bottom = np.array([landmarks[14].x*w, landmarks[14].y*h])
    width = np.linalg.norm(mouth_left - mouth_right)
    height = np.linalg.norm(mouth_top - mouth_bottom)
    return width / max(height, 1.0)

def bounding_rect_from_landmarks(landmarks, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)

def boxes_overlap(face_box, hand_box):
    x1,y1,x2,y2 = [int(v) for v in face_box]
    hx1,hy1,hx2,hy2 = [int(v) for v in hand_box]
    intersect = max(0, min(x2,hx2)-max(x1,hx1)) * max(0, min(y2,hy2)-max(y1,hy1))
    hand_area = max(1, (hx2-hx1)*(hy2-hy1))
    return intersect / hand_area > 0.25

def play_alert(path):
    def _play(): 
        try: playsound(path)
        except: pass
    threading.Thread(target=_play, daemon=True).start()

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.4, min_tracking_confidence=0.4)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4)

# SETUP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
screen_width, screen_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

EAR_BASELINE_SAMPLES = []
DYNAMIC_THRESHOLD = 0.25
frame_count = detection_failures = consecutive_drowsy = 0
last_drowsy_alert = last_distract_alert = 0

# Sounds
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DROWSY_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "drowsiness_alert.wav")
DISTRACT_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "distraction_alert.wav")

# CSV
csv_path = "phase1_elite.csv"
csv_file = open(csv_path, "w", newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["ts","frame","status","drowsy","yawn","left","right","up","down","blocked","sleep_s","ear","mar","yaw","pitch","fps","bright"])
print(f"🚀 PHASE 1 ELITE CSV: {csv_path}")

window_name = "🚗 ELITE DMS PHASE 1 - ESC/Q"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_width, screen_height)
cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

print("🎉 PHASE 1 READY!")
print(" TEST: 1.YAWN(BIG) 2.LEFT 3.RIGHT 4.DOWN 5.HAND_FACE 6.CLOSE_EYES")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    start = time.time()
    frame_count += 1
    
    frame = cv2.resize(frame, (screen_width, screen_height))
    frame = preprocess_ultra_lowlight(frame)
    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results_face = face_mesh.process(frame_rgb)
    results_hands = hands.process(frame_rgb)
    
    # PHASE 1 FEATURES
    drowsy = yawning = head_left = head_right = head_up = head_down = obstructed = False
    ear_val = mar_val = head_yaw = head_pitch = 0.0
    FACE_BOX = (0,0,0,0)
    
    if results_face.multi_face_landmarks:
        face_lms = results_face.multi_face_landmarks[0].landmark
        mp_drawing.draw_landmarks(frame, results_face.multi_face_landmarks[0],
                                mp_face_mesh.FACEMESH_TESSELATION,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(200,200,255), thickness=1))
        
        #  FACE BOX (Blue)
        xs = [int(lm.x * iw) for lm in face_lms]
        ys = [int(lm.y * ih) for lm in face_lms]
        FACE_BOX = (min(xs)-30, min(ys)-30, max(xs)+30, max(ys)+30)
        cv2.rectangle(frame, FACE_BOX, (255,0,0), 2)
        
        #  YAWN FIXED (YOUR 4 POINTS + RED DOTS)
        mar_val = mouth_aspect_ratio(face_lms, iw, ih)
        yawning = mar_val < 1.8  # YOUR PERFECT THRESHOLD
        for i in [61, 291, 13, 14]:  # Left, Right, Top, Bottom
            x, y = int(face_lms[i].x*iw), int(face_lms[i].y*ih)
            cv2.circle(frame, (x,y), 6, (0,0,255), -1)
        cv2.putText(frame, f"MAR:{mar_val:.2f}", (xs[0]+10, ys[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        #  HEAD POSE (GREEN LINE + MAGENTA NOSE)
        nose_x, nose_y = face_lms[1].x, face_lms[1].y
        left_ear_x = face_lms[234].x
        right_ear_x = face_lms[454].x
        chin_y = face_lms[152].y
        
        head_yaw = (nose_x - (left_ear_x + right_ear_x)/2) * 300  # PERFECTED
        head_pitch = (nose_y - chin_y) * 150                      # PERFECTED
        
        head_left = head_yaw < -20
        head_right = head_yaw > 20
        head_up = head_pitch < -30
        head_down = head_pitch > 30
        
        nose_pt = (int(nose_x*iw), int(nose_y*ih))
        chin_pt = (int(face_lms[152].x*iw), int(chin_y*ih))
        cv2.line(frame, nose_pt, chin_pt, (0,255,0), 3)
        cv2.circle(frame, nose_pt, 8, (255,0,255), -1)
        
        cv2.putText(frame, f"Y:{head_yaw:.0f} P:{head_pitch:.0f}", (xs[0]+10, ys[0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        
        #  DROWSY (YOUR MODULE)
        ear_left = eye_aspect_ratio(face_lms, LEFT_EYE_IDX, iw, ih)
        ear_right = eye_aspect_ratio(face_lms, RIGHT_EYE_IDX, iw, ih)
        ear_val = (ear_left + ear_right) / 2.0
        drowsy = ear_val < 0.25
        
        EAR_BASELINE_SAMPLES.append(ear_val)
        if len(EAR_BASELINE_SAMPLES) > 500: EAR_BASELINE_SAMPLES.pop(0)
        if len(EAR_BASELINE_SAMPLES) >= 50:
            DYNAMIC_THRESHOLD = max(0.18, np.mean(EAR_BASELINE_SAMPLES[-100:]) * 0.75)
        
        if drowsy: 
            consecutive_drowsy += 1
        else: 
            consecutive_drowsy = 0
    
    #  HANDS (YOUR detector + BLOCK)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks[:2]:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,200), thickness=2))
            
            if results_face.multi_face_landmarks:
                face_lms = results_face.multi_face_landmarks[0].landmark
                using_your_detector = hand_near_face(face_lms, hand_landmarks.landmark, iw, ih)
                HAND_BOX = bounding_rect_from_landmarks(hand_landmarks.landmark, iw, ih)
                hx1, hy1, hx2, hy2 = max(0, HAND_BOX[0]-15), max(0, HAND_BOX[1]-15), min(iw, HAND_BOX[2]+15), min(ih, HAND_BOX[3]+15)
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0,255,0), 2)
                obstructed = boxes_overlap(FACE_BOX, (hx1, hy1, hx2, hy2)) or using_your_detector
    
    #  STATUS
    status_parts = []
    if drowsy: status_parts.append(f"DROWSY({ear_val:.2f})")
    if yawning: status_parts.append(f"YAWN({mar_val:.2f})")
    if head_left: status_parts.append("LEFT")
    if head_right: status_parts.append("RIGHT")
    if head_up: status_parts.append("UP")
    if head_down: status_parts.append("DOWN")
    if obstructed: status_parts.append("BLOCKED")
    if consecutive_drowsy >= 150: status_parts.append("SLEEP")
    
    status = " + ".join(status_parts) if status_parts else f"NORMAL E:{ear_val:.2f}"
    color = (0,0,255) if consecutive_drowsy >= 150 else (0,165,255) if status_parts else (0,255,0)
    
    #  DISPLAY
    cv2.putText(frame, status, (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"EAR:{ear_val:.2f} MAR:{mar_val:.2f} T:{DYNAMIC_THRESHOLD:.2f}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    fps = int(1.0/(time.time()-start+1e-6))
    brightness_pct = int(np.mean(frame)/2.55)
    cv2.putText(frame, f"FPS:{fps} B:{brightness_pct}%", (iw-220,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    if not results_face.multi_face_landmarks: detection_failures += 1
    cv2.putText(frame, f"Loss:{detection_failures}", (20, ih-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    
    # CSV
    ts = time.strftime('%H:%M:%S')
    csv_writer.writerow([ts, frame_count, status, drowsy, yawning, head_left, head_right, head_up, head_down, obstructed, 
                        consecutive_drowsy//30, round(ear_val,3), round(mar_val,3), round(head_yaw,2), round(head_pitch,2), 
                        fps, brightness_pct])
    csv_file.flush()
    
    # ALERTS
    now = time.time()
    if (drowsy or yawning) and now-last_drowsy_alert > 3: 
        play_alert(DROWSY_ALERT_PATH); last_drowsy_alert = now
    if obstructed and now-last_distract_alert > 3: 
        play_alert(DISTRACT_ALERT_PATH); last_distract_alert = now
    
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
print(f"\n🎉 PHASE 1 ELITE COMPLETE! {frame_count} frames → phase1_elite.csv")
print("✅ 6 FEATURES + LOW LIGHT + YOUR MODULES = PRODUCTION READY!")


































# """
# main_overlay.py - Enhanced overlay (face wireframe, colored hand/keypoints, bounding boxes)
# Creates same behavior as original main.py but with improved visual overlay.
# Run: pip install opencv-python mediapipe numpy playsound ultralytics
#       python main_overlay.py
# """
# import os, time, threading, cv2, numpy as np
# import mediapipe as mp
# from playsound import playsound

# # import helper funcs from original project if available
# try:
#     from drowsiness_detector import eye_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX
# except Exception as e:
#     # fallback: simple EAR compute using landmark indices (same indices expected)
#     print("Warning: couldn't import drowsiness_detector, using local EAR. Error:", e)
#     LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
#     RIGHT_EYE_IDX = [362, 387, 385, 263, 373, 380]
#     def eye_aspect_ratio(landmarks, eye_idx, w, h):
#         pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_idx]
#         A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
#         B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
#         C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
#         if C==0: return 0.0
#         return (A+B) / (2.0*C)

# try:
#     from distraction_detector import hand_near_face
# except Exception as e:
#     print("Warning: couldn't import distraction_detector.hand_near_face. Error:", e)
#     def hand_near_face(face_lms, hand_lms, w, h):
#         # naive proximity: compare distance between nose (1) and hand wrist (0)
#         try:
#             fx = int(face_lms[1].x * w); fy = int(face_lms[1].y * h)
#             hx = int(hand_lms[0].x * w); hy = int(hand_lms[0].y * h)
#             return np.hypot(fx-hx, fy-hy) < 120
#         except:
#             return False

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DROWSY_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "drowsiness_alert.wav")
# DISTRACT_ALERT_PATH = os.path.join(BASE_DIR, "sounds", "distraction_alert.wav")

# # MediaPipe initializations
# mp_face_mesh = mp.solutions.face_mesh
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
#                                   refine_landmarks=True, min_detection_confidence=0.5,
#                                   min_tracking_confidence=0.5)
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
#                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

# def euclidean(a,b): return np.linalg.norm(np.array(a)-np.array(b))

# def compute_ear_from_landmarks(landmarks, eye_idx, w, h):
#     pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_idx]
#     A = euclidean(pts[1], pts[5])
#     B = euclidean(pts[2], pts[4])
#     C = euclidean(pts[0], pts[3])
#     return 0.0 if C==0 else (A+B)/(2.0*C)

# def bounding_rect_from_landmarks(landmarks, w, h):
#     xs = [int(lm.x * w) for lm in landmarks]
#     ys = [int(lm.y * h) for lm in landmarks]
#     return min(xs), min(ys), max(xs), max(ys)

# def play_alert(path):
#     # non-blocking playsound
#     def _play():
#         try:
#             playsound(path)
#         except:
#             pass
#     t = threading.Thread(target=_play, daemon=True)
#     t.start()

# # Video source: try sample video in project, fallback to webcam
# video_path = os.path.join(BASE_DIR, "sample_videos", "video.mp4")
# VIDEO_SRC = video_path if os.path.exists(video_path) else 0

# cap = cv2.VideoCapture(VIDEO_SRC)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# last_drowsy_alert = 0
# last_distract_alert = 0
# DROWSY_COOLDOWN = 5.0
# DISTRACT_COOLDOWN = 5.0

# window_name = "Driver Monitor - Press Q to quit"

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     start = time.time()
#     frame = cv2.flip(frame, 1)
#     ih, iw = frame.shape[:2]
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results_face = face_mesh.process(frame_rgb)
#     results_hands = hands.process(frame_rgb)

#     drowsy = False
#     distracted = False
#     ear_val = None

#     if results_face.multi_face_landmarks:
#         face_lms = results_face.multi_face_landmarks[0].landmark
#         # draw white wireframe mesh (tesselation + contours)
#         mp_drawing.draw_landmarks(frame, results_face.multi_face_landmarks[0],
#                                   mp_face_mesh.FACEMESH_TESSELATION,
#                                   landmark_drawing_spec=None,
#                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
#         mp_drawing.draw_landmarks(frame, results_face.multi_face_landmarks[0],
#                                   mp_face_mesh.FACEMESH_CONTOURS,
#                                   landmark_drawing_spec=None,
#                                   connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
#         # EAR using helper if available
#         try:
#             ear_left = eye_aspect_ratio(face_lms, LEFT_EYE_IDX, iw, ih)
#             ear_right = eye_aspect_ratio(face_lms, RIGHT_EYE_IDX, iw, ih)
#         except:
#             ear_left = compute_ear_from_landmarks(face_lms, LEFT_EYE_IDX, iw, ih)
#             ear_right = compute_ear_from_landmarks(face_lms, RIGHT_EYE_IDX, iw, ih)
#         ear_val = (ear_left + ear_right) / 2.0
#         if ear_val < 0.22:
#             drowsy = True

#         # face bounding box
#         xs = [int(lm.x * iw) for lm in face_lms]
#         ys = [int(lm.y * ih) for lm in face_lms]
#         fx1, fy1, fx2, fy2 = max(0, min(xs)-20), max(0, min(ys)-20), min(iw, max(xs)+20), min(ih, max(ys)+20)
#         # Hands
#     if results_hands.multi_hand_landmarks:
#         for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
#             # draw red dots for keypoints
#             for lm in hand_landmarks.landmark:
#                 x, y = int(lm.x*iw), int(lm.y*ih)
#                 cv2.circle(frame, (x,y), 4, (0,0,255), -1)
#             # draw connections in white
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                       landmark_drawing_spec=None,
#                                       connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=0))
#             # bounding box for hand
#             hx1, hy1, hx2, hy2 = bounding_rect_from_landmarks(hand_landmarks.landmark, iw, ih)
#             pad=10
#             hx1, hy1 = max(0, hx1-pad), max(0, hy1-pad)
#             hx2, hy2 = min(iw, hx2+pad), min(ih, hy2+pad)
#             color = (255,0,0) if idx==0 else (0,0,255)
#             cv2.putText(frame, f"Hand {idx+1}", (hx1, hy1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             # check hand near face using project's helper if available
#             if results_face.multi_face_landmarks:
#                 if hand_near_face(face_lms, hand_landmarks.landmark, iw, ih):
#                     distracted = True

#     # Alerts
#     if drowsy and time.time()-last_drowsy_alert > DROWSY_COOLDOWN:
#         play_alert(DROWSY_ALERT_PATH)
#         last_drowsy_alert = time.time()
#     if distracted and time.time()-last_distract_alert > DISTRACT_COOLDOWN:
#         play_alert(DISTRACT_ALERT_PATH)
#         last_distract_alert = time.time()

#     # Overlay texts
#     if ear_val is not None:
#         cv2.putText(frame, f"EAR: {ear_val:.3f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
#     status = "Normal"
#     color = (0,255,0)
#     if drowsy and distracted:
#         status = "DROWSY + DISTRACTED"
#         color = (0,0,255)
#     elif drowsy:
#         status = "DROWSY"
#         color = (0,0,255)
#     elif distracted:
#         status = "DISTRACTED (Device)"
#         color = (0,165,255)
#     cv2.putText(frame, status, (20,90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 6)

#     # FPS
#     fps = int(1.0/(time.time()-start+1e-6))
#     cv2.putText(frame, f"FPS: {fps}", (iw-140,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200),2)

#     cv2.imshow(window_name, frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# face_mesh.close()
# hands.close()
