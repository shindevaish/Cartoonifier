import cv2
import numpy as np


def resize_image(img, max_size=800):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def pencil_sketch(img, blur_size=21):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 1. make it gray
    inverted = 255 - gray # 2. image negative
    blur = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0) # 3. gaussian blur 
    sketch = cv2.divide(gray, 255 - blur, scale=255) # 4. color dodge
    return sketch


def color_pencil(img, saturation=1.5, blur_size=21):
    sketch = pencil_sketch(img, blur_size)
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR) / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR) / 255.0

    result = vivid * sketch
    return (result * 255).astype(np.uint8)


def cartoonify(img, iterations=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 2)

    color = img
    for _ in range(iterations):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    return cv2.bitwise_and(color, color, mask=edges)
