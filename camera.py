from __future__ import annotations

import cv2
import mediapipe as mp

from config import DETECTION_CONFIDENCE, TRACKING_CONFIDENCE

hands_api = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils


def open_camera(max_indexes: int = 5):
    for index in range(max_indexes):
        camera = cv2.VideoCapture(index)

        if camera.isOpened():
            ok, _ = camera.read()
            if ok:
                print(f"Camera encontrada: indice {index}")
                return camera

        camera.release()

    return None


def create_hands_model(max_hands: int):
    return hands_api.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        model_complexity=1,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )


def get_visible_hands(result) -> list[tuple[str, object]]:
    if not result.multi_hand_landmarks or not result.multi_handedness:
        return []

    return [
        (side.classification[0].label, hand)
        for hand, side in zip(result.multi_hand_landmarks, result.multi_handedness)
    ]


def detect_hands(frame, hands_model):
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    hands = get_visible_hands(hands_model.process(image_rgb))

    for _, hand in hands:
        drawing_utils.draw_landmarks(frame, hand, hands_api.HAND_CONNECTIONS)

    return frame, hands