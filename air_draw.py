import mediapipe as mp
from mediapipe.tasks import python

MODEL_PATH = "models/hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

HandLandmarker = mp.tasks.vision.HandLandmarker

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

hand_landmarker = HandLandmarker.create_from_options(options)
