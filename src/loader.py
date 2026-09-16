from pathlib import Path

from PIL import Image


def load_image(source: Path | str) -> Image.Image:
    try:
        return Image.open(source).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {source}")
