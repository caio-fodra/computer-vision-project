from __future__ import annotations

from typing import Optional

import numpy as np

NUM_LANDMARKS = 21
FEATURE_SIZE = NUM_LANDMARKS * 3


def extract_landmarks(hand, handedness: Optional[str] = None) -> np.ndarray:
    landmarks = np.array(
        [[point.x, point.y, point.z] for point in hand.landmark],
        dtype=np.float32,
    )
    landmarks -= landmarks[0]

    if handedness == "Left":
        landmarks[:, 0] *= -1.0

    scale = float(np.max(np.abs(landmarks)))
    if scale > 0:
        landmarks /= scale

    return landmarks.flatten()