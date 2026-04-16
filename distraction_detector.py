"""
distraction_detector.py
Contains functions to detect device usage / hand-near-face heuristics using MediaPipe Hands.
"""
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def hand_near_face(face_landmarks, hand_landmarks, image_w, image_h, pad_x_factor=0.2, pad_y_factor=0.4):
    xs = [lm.x * image_w for lm in face_landmarks]
    ys = [lm.y * image_h for lm in face_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * pad_x_factor
    pad_y = (y_max - y_min) * pad_y_factor
    x_min -= pad_x; x_max += pad_x; y_min -= pad_y; y_max += pad_y
    for lm in hand_landmarks:
        x, y = lm.x * image_w, lm.y * image_h
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    return False
