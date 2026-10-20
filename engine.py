"""engine.py - the image-processing part (no UI here)."""
import cv2
import numpy as np


def resize_image(img, max_size=800):
    """Shrink big photos so the filters run fast."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


# ---------- Style 1: Black & White pencil sketch ----------
def pencil_sketch(img, blur_size=21):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # 1. make it gray
    inverted = 255 - gray                                  # 2. make the negative
    blur = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)  # 3. blur it
    sketch = cv2.divide(gray, 255 - blur, scale=255)       # 4. color dodge
    return sketch


# ---------- Style 2: Color pencil sketch ----------
def color_pencil(img, saturation=1.5, blur_size=21):
    # sketch lines (made 3-channel so it can multiply with a color image)
    sketch = pencil_sketch(img, blur_size)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR) / 255.0

    # make colors more vivid using HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR) / 255.0

    # multiply colors with sketch lines
    result = vivid * sketch
    return (result * 255).astype(np.uint8)


# ---------- Style 3: Comic cartoon ----------
def cartoonify(img, iterations=5):
    # a) black outlines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)

    # b) flat smooth colors
    color = img
    for _ in range(iterations):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    # c) put the outlines on the flat colors
    return cv2.bitwise_and(color, color, mask=edges)
