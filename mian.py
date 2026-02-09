import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from collections import deque

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

MODEL_PATH = 'hand_landmarker.task'
SMOOTH_VALUES = 8
vol_history = deque(maxlen=SMOOTH_VALUES)

BaseOptions = mp.tasks.BaseOptions
HandLandmark = vision.HandLandmarker
HandLandmarkOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

with HandLandmark.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    pTime = 0
    while True:
        success, frame = cap.read()
        if not success: break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        frame_timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # We need the coordinates of the thumb tip (4) and index finger tip (8) to calculate the distance between them.
                # In new version of mediapipe, the hand landmarks are normalized to [0.0, 1.0] range.
                h, w, _ = frame.shape

                # Point 4 (Thumb tip)
                x1 = int(hand_landmarks[4].x * w)
                y1 = int(hand_landmarks[4].y * h)

                # Point 8 (Index finger tip)
                x2 = int(hand_landmarks[8].x * w)
                y2 = int(hand_landmarks[8].y * h)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Calculate the center point between the thumb and index finger
                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                length = math.hypot(x2 - x1, y2 - y1) # Euclidean distance between the thumb and index finger

                current_vol = np.interp(length, [20, 200], [minVol, maxVol])
                vol_history.append(current_vol)
                smoothed_vol = sum(vol_history) / len(vol_history)

                volBar = np.interp(length, [30, 200], [400, 150])
                volPer = np.interp(length, [30, 200], [0, 100])

                volume.SetMasterVolumeLevel(smoothed_vol, None)
                if length < 30:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

                cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Volume Control (New API)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

