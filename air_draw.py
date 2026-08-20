import cv2
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

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pixel_x = int(hand[8].x * frame.shape[1])
            pixel_y = int(hand[8].y * frame.shape[0])

            cv2.circle(frame, (pixel_x, pixel_y), 10, (255, 0, 0), -1)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break