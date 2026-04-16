# visualization_overlay_rectified.py
# Requires: mediapipe, opencv-python
# pip install mediapipe opencv-python

import cv2
import mediapipe as mp
import time
import math
import numpy as np

# --- Helpers ---------------------------------------------------------
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# FaceMesh eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    # landmarks: list of normalized face landmarks (x,y)
    # returns EAR float
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * image_w), int(lm.y * image_h)))
    # vertical distances
    A = math.dist(pts[1], pts[5])
    B = math.dist(pts[2], pts[4])
    # horizontal
    C = math.dist(pts[0], pts[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def draw_thick_text(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=1.5, color=(0,140,255), thickness=8, outline_color=(0,0,0)):
    # Draw outline by drawing text multiple times with larger thickness
    x,y = org
    # outline (draw slightly offset thicker black text first)
    cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 4, cv2.LINE_AA)
    # main text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_glowing_text(img, text, org, color, font_scale=2.0):
    # simulate glow by adding a blurred colored blob behind text
    overlay = img.copy()
    alpha = 0.7
    # get text size
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)
    x,y = org
    # bounding box background (bigger than text)
    pad = 20
    x0 = max(0, x - pad)
    y0 = max(0, y - h - pad)
    x1 = min(img.shape[1], x + w + pad)
    y1 = min(img.shape[0], y + pad)
    # colored rectangle behind text (will be blurred)
    cv2.rectangle(overlay, (x0,y0), (x1,y1), color, -1)
    # blur to create glow
    blurred = cv2.GaussianBlur(overlay, (31,31), 0)
    cv2.addWeighted(blurred, 0.6, img, 0.4, 0, img)
    # now draw thick outlined text on top
    draw_thick_text(img, text, (x, y), font_scale=font_scale, color=(255,255,255), thickness=6, outline_color=(0,0,0))

# --- Main loop ------------------------------------------------------
def main(camera_idx=0):
    cap = cv2.VideoCapture(camera_idx)
    # optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face.FaceMesh(static_image_mode=False,
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image_h, image_w = frame.shape[:2]

            # flip for selfie view (optional)
            frame = cv2.flip(frame, 1)

            # convert for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # face mesh
            face_results = face_mesh.process(rgb)
            # hands
            hand_results = hands.process(rgb)

            # create transparent overlay for drawing (keeps original camera visible)
            overlay = frame.copy()

            # --- draw face mesh (wireframe) ---
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # draw wireframe lines
                    for i, lm in enumerate(face_landmarks.landmark):
                        x = int(lm.x * image_w)
                        y = int(lm.y * image_h)
                        # small points for dense mesh (optional)
                        # cv2.circle(overlay, (x,y), 1, (200,200,200), -1)

                    # draw mesh edges using mp drawing utilities or custom lines for a wireframe:
                    mp.solutions.drawing_utils.draw_landmarks(
                        overlay,
                        face_landmarks,
                        mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles
                                                   .get_default_face_mesh_tesselation_style()
                    )

            # --- draw hands (skeleton) ---
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # draw connections
                    mp.solutions.drawing_utils.draw_landmarks(
                        overlay,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )
                    # emphasize tips with red dots
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * image_w)
                        y = int(lm.y * image_h)
                        cv2.circle(overlay, (x,y), 4, (0,0,255), -1)

            # Mix overlay onto frame (alpha blending)
            alpha = 0.9
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # --- compute EAR and status label ---
            ear_val = 0.0
            status_text = "Normal"
            status_color = (0, 200, 0)  # green default

            if face_results.multi_face_landmarks:
                lm_list = face_results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(lm_list, LEFT_EYE, image_w, image_h)
                right_ear = eye_aspect_ratio(lm_list, RIGHT_EYE, image_w, image_h)
                ear_val = (left_ear + right_ear) / 2.0
                # thresholding (example)
                if ear_val < 0.25:
                    status_text = "DROWSY"
                    status_color = (0, 140, 255)  # orange-ish
                else:
                    status_text = "Normal"
                    status_color = (0, 200, 0)

            # --- FPS calculation ---
            cur_time = time.time()
            dt = cur_time - prev_time if cur_time - prev_time > 0 else 1e-6
            fps = 1.0 / dt
            prev_time = cur_time

            # --- draw EAR top-left ---
            ear_str = f"EAR: {ear_val:.3f}"
            cv2.putText(frame, ear_str, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            # --- draw big status top-left (glow + outline) ---
            # choose position and text
            big_text = None
            # Example: if you detect phone in hand or hand raised, you might set "DISTRACTED (Device)".
            # Here we show status_text; modify logic to detect device usage as you like.
            if status_text != "Normal":
                big_text = f"{status_text} (Device)"  # sample
            else:
                big_text = "Normal"

            # draw glowing orange when distracted, green when normal
            if "DROWSY" in big_text or "DISTRACTED" in big_text:
                glow_color = (0,140,255)  # orange-ish
            else:
                glow_color = (0,200,0)  # green

            # position near top center-left
            text_x = 60
            text_y = 140
            draw_glowing_text(frame, big_text, (text_x, text_y), glow_color, font_scale=2.0)

            # --- draw FPS top-right with white text and small shadow ---
            fps_text = f"FPS: {int(fps)}"
            (fw, fh), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            fps_x = frame.shape[1] - fw - 20
            fps_y = 40
            # shadow
            cv2.putText(frame, fps_text, (fps_x+2,fps_y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, fps_text, (fps_x,fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            # --- show on screen ---
            cv2.imshow("Driver Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        hands.close()

if __name__ == "__main__":
    main()
