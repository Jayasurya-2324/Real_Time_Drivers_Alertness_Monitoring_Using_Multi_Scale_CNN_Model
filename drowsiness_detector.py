"""
drowsiness_detector.py
Provides EAR computation and drowsiness detection helper.
"""
import numpy as np

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * image_w, lm.y * image_h]))
    A = np.linalg.norm(coords[1] - coords[5])
    B = np.linalg.norm(coords[2] - coords[4])
    C = np.linalg.norm(coords[0] - coords[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear
