"""ChromaSketch engine: classical-CV non-photorealistic rendering (CPU only)."""
import time
import cv2
import numpy as np


def resize_max(bgr, max_side=1000):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    return bgr if s >= 1 else cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _odd(k):
    return int(k) | 1


def paper_grain(shape, strength=0.0, seed=7):
    """Multiplicative paper-tooth texture (creative extra). Returns float map ~1.0."""
    if strength <= 0:
        return np.ones(shape[:2], np.float32)
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, shape[:2]).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.8)
    n /= (np.abs(n).max() + 1e-6)
    return 1.0 - strength * 0.5 * (n + 1) * 0.5


# ---------------- Module 1: B&W pencil ----------------
def pencil_sketch(bgr, ksize=21, grain=0.0, steps=None):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (_odd(ksize), _odd(ksize)), 0)
    # Color dodge: gray / (255 - blur) * 255
    sketch = cv2.divide(gray, 255 - blur, scale=255)
    if grain > 0:
        sketch = np.clip(sketch * paper_grain(sketch.shape, grain), 0, 255).astype(np.uint8)
    if steps is not None:
        steps.update({"1 · Grayscale": gray, "2 · Inverted": inv,
                      "3 · Gaussian blur": blur, "4 · Color dodge": sketch})
    return sketch


# ---------------- Module 2: Color pencil ----------------
def color_pencil(bgr, sat=1.5, ksize=21, stroke_gamma=1.0, grain=0.0, steps=None):
    sketch = pencil_sketch(bgr, ksize, grain)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
    mask = (sketch.astype(np.float32) / 255.0) ** stroke_gamma
    mask3 = cv2.merge([mask, mask, mask])
    out = np.clip(vivid * mask3 * 255, 0, 255).astype(np.uint8)
    if steps is not None:
        steps.update({"1 · Saturation ×%.1f" % sat: (vivid * 255).astype(np.uint8),
                      "2 · Pencil mask": (mask3 * 255).astype(np.uint8),
                      "3 · Composite": out})
    return out


# ---------------- Module 3: Cartoonifier ----------------
def cartoonify(bgr, iterations=5, block=9, c=2, sigma=75, posterize=0, steps=None):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, _odd(block), c)
    # Bilateral filtering is costly: run it at half-res for big images, then upsample.
    small = bgr
    big = max(bgr.shape[:2]) > 700
    if big:
        small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    color = small
    for _ in range(int(iterations)):
        color = cv2.bilateralFilter(color, 9, sigma, sigma)
    if big:
        color = cv2.resize(color, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    if posterize and posterize > 1:           # creative extra: flat poster-like color bands
        step = 256 // int(posterize)
        color = (color // step * step + step // 2).astype(np.uint8)
    out = cv2.bitwise_and(color, color, mask=edges)
    if steps is not None:
        steps.update({"1 · Adaptive edges": edges, "2 · Bilateral ×%d" % iterations: color,
                      "3 · Bitwise AND": out})
    return out


# ---------------- Creative extra: Ink & Wash fusion ----------------
def ink_wash(bgr, wash=0.6, **kw):
    """Cartoon colour wash under pencil line-work (blend of Modules 1 and 3)."""
    base = cartoonify(bgr, kw.get("iterations", 3), posterize=kw.get("posterize", 6))
    line = cv2.cvtColor(pencil_sketch(bgr, 21), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255
    soft = cv2.addWeighted(base, wash, np.full_like(base, 255), 1 - wash, 0).astype(np.float32)
    return np.clip(soft * line, 0, 255).astype(np.uint8)


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, (time.perf_counter() - t) * 1000
