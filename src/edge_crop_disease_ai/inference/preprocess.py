"""Image preprocessing helpers shared across inference backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_class_names(config: dict[str, Any]) -> list[str]:
    """Load class names from the label map file written during dataset setup."""
    label_map_path = Path(config["inference"]["class_names_path"]).expanduser()
    if not label_map_path.exists():
        raise FileNotFoundError(f"Class names file not found: {label_map_path}")

    with label_map_path.open("r", encoding="utf-8") as handle:
        class_names = [line.strip() for line in handle if line.strip()]

    if not class_names:
        raise ValueError(f"No class names found in label map: {label_map_path}")
    return class_names


def load_and_preprocess_image(image_path: str | Path, config: dict[str, Any]) -> Any:
    """Load an image from disk and transform it into model input format.

    Args:
        image_path: Path to the input image.
        config: Parsed project configuration.

    Returns:
        Preprocessed tensor or array suitable for model inference.

    """
    import numpy as np
    from PIL import Image

    image_file = Path(image_path).expanduser()
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")

    image_size = int(config["data"]["image_size"])
    channels = int(config["data"].get("channels", 3))

    image = Image.open(image_file).convert("RGB" if channels == 3 else "L")
    image = image.resize((image_size, image_size))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    if channels == 1:
        image_array = np.expand_dims(image_array, axis=-1)

    image_array = np.expand_dims(image_array, axis=0)
    return image_array.astype(np.float32)


def build_topk_predictions(
    probabilities: np.ndarray,
    class_names: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Convert class probabilities into ranked prediction records."""
    import numpy as np

    squeezed = np.asarray(probabilities).reshape(-1)
    if squeezed.size != len(class_names):
        raise ValueError(
            "Probability vector size does not match number of class names: "
            f"{squeezed.size} vs {len(class_names)}"
        )

    top_k = max(1, min(int(top_k), len(class_names)))
    top_indices = np.argsort(squeezed)[::-1][:top_k]
    return [
        {
            "class_index": int(index),
            "class_name": class_names[int(index)],
            "confidence": float(squeezed[int(index)]),
        }
        for index in top_indices
    ]
