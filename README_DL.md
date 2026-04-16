# Deep-Learning Enhanced Version (DL edition)

This added version includes:
- `main_dl.py` — integrates optional CNN eye-state model and optional YOLOv8 phone detection.
- `train_eye_model.py` — training scaffold to create models/eye_state.h5 and TFLite model.
- Instructions below.

## How to run
1. Install dependencies (recommended):
   ```bash
   pip install -r requirements.txt
   pip install tensorflow   # optional, for training/using Keras model
   pip install ultralytics  # optional, for YOLO phone detection
   ```
2. If you have a pre-trained eye-state model (models/eye_state.h5 or .tflite), place it in `models/`.
3. Run the DL-enabled main:
   ```bash
   python main_dl.py --source webcam
   # or
   python main_dl.py --source sample --video_path sample_videos/video.mp4
   ```

## To train eye-state model
Prepare dataset as:
```
dataset/
  open/
  closed/
```
Then run:
```
python train_eye_model.py --dataset dataset --epochs 15
```

Notes:
- If ultralytics / YOLO is not available, the code falls back to the existing hand_near_face heuristic.
- The DL model improves robustness under glasses and varied lighting.
