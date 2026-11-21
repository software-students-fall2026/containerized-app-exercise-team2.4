import io
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


def detect_hand(image_bytes):
    """Return 21 hand landmarks, or None."""

    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    image_array = np.asarray(image)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_array,
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    return result.hand_landmarks[0]
