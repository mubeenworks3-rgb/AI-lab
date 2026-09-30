from pathlib import Path

import numpy as np
from PIL import Image


def image_to_features(path, size=(64, 64)):
    """Resize an image, convert it to grayscale, and return a flat pixel vector."""
    with Image.open(Path(path)) as image:
        grayscale = image.convert("L").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(grayscale, dtype=np.uint8).reshape(-1)
