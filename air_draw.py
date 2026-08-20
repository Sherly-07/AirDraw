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

prev_x = None
prev_y = None

canvas = None

while True:
    success, frame = cap.read()

    if canvas is None:
        canvas = frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            pixel_x = int(hand[8].x * frame.shape[1])
            pixel_y = int(hand[8].y * frame.shape[0])

            cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 255, 0), -1)

            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (pixel_x, pixel_y), (0, 255, 0), 5)

            prev_x = pixel_x
            prev_y = pixel_y    

    frame = cv2.addWeighted(frame, 1, canvas, 1, 0)        

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break