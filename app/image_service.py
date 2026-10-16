"""Decode images, normalize orientation, and draw annotations."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import config
from .models import Detection


def read_image(path: Path) -> Image.Image:
    path = Path(path)
    if path.suffix.lower() not in config.IMAGE_SUFFIXES:
        raise ValueError("รองรับเฉพาะ JPG, JPEG และ PNG")
    try:
        with Image.open(path) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("เนื้อหาไฟล์ไม่ใช่ JPEG หรือ PNG")
            return ImageOps.exif_transpose(source).convert("RGB")
    # Ultralytics patches Pillow and may try an optional HEIF plugin after a decode failure.
    except (OSError, ImportError, Image.DecompressionBombError) as exc:
        raise ValueError(f"ไม่สามารถอ่านไฟล์ภาพ: {path.name}") from exc


def annotate(image: Image.Image, detections: tuple[Detection, ...]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    size = max(12, min(28, min(image.size) // 35))
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for detection in detections:
        detection.validate(*image.size)
        color = config.class_color(detection.class_id)
        draw.rectangle((detection.x1, detection.y1, detection.x2, detection.y2),
                       outline=color, width=max(2, min(image.size) // 250))
        text = f"{detection.class_name} {detection.confidence:.2f}"
        bounds = draw.textbbox((0, 0), text, font=font)
        width, height = bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 8
        x, y = max(0, min(detection.x1, canvas.width - width)), max(0, detection.y1 - height)
        draw.rectangle((x, y, x + width, y + height), fill=color)
        draw.text((x + 4, y + 4 - bounds[1]), text, fill="white", font=font)
    return canvas
