# ChromaSketch — Artistic Cartoonifier & Color Sketch Generator

> **Non-photorealistic rendering and image stylization using classical computer vision.**
> Turn any photo into a pencil sketch, a vivid colour-pencil drawing or a comic cartoon — in real time, on CPU only, with no neural networks.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Overview

Modern stylization approaches (GANs, Neural Style Transfer) need GPUs and large trained models. **ChromaSketch** shows that convincing artistic effects can be produced with classical image-processing operators — spatial filtering, colour-space manipulation and pixel-matrix arithmetic — running in tens of milliseconds on a laptop CPU.

## Features

| Style | Colour space | Core operator | Output |
|---|---|---|---|
| **Pencil Sketch** | Grayscale | Colour dodge (`cv2.divide`) | Monochrome graphite drawing |
| **Color Pencil** | BGR + HSV | Saturation boost + matrix multiplication | Hand-coloured lead drawing |
| **Cartoonifier** | BGR + Gray | Bilateral filter (×5) + adaptive threshold | Flat comic-book art |
| **Ink & Wash** *(bonus)* | BGR + Gray | Cartoon wash × pencil line-work | Watercolour-and-pencil look |

**UI extras:** live parameter sliders · webcam input · "all styles" grid · before/after reveal slider · pipeline inspector (see every intermediate step) · render-time / FPS metrics · PNG download · paper-grain texture · poster colour bands.

## Methodology

### 1. Pencil Sketch
1. Convert BGR → grayscale `I_gray`
2. Invert: `I_inv = 255 − I_gray`
3. Gaussian blur (default 21×21): `I_blur`
4. Colour-dodge blend:

```
Sketch = I_gray / (255 − I_blur) × 255
```

Flat regions give a ratio of ≈1 (white paper); edges give <1 (dark strokes).

### 2. Color Pencil
1. Convert to HSV
2. Multiply Saturation by 1.5 (hue untouched)
3. Convert back to BGR and normalise to `[0, 1]`
4. Multiply by the 3-channel pencil mask → colour with graphite shading

### 3. Cartoonifier
1. **Edges:** grayscale → 5×5 median blur → adaptive (mean) threshold → black outlines
2. **Surfaces:** 5 iterations of bilateral filter → flat colour, sharp boundaries
3. **Composite:** `cv2.bitwise_and(color, color, mask=edges)`

### 4. Ink & Wash *(bonus)*
Posterised cartoon colour is faded toward white with `cv2.addWeighted`, then multiplied by the pencil sketch.

> 📖 A line-by-line explanation of every function is in [`CODE_EXPLANATION.txt`](CODE_EXPLANATION.txt).

## Architecture

```
┌──────────────┐    ┌───────────────────────────────────────────┐    ┌─────────────┐
│  Input       │    │              engine.py                    │    │  Output     │
│  • Upload    │───▶│  resize_max ─▶ style pipeline ─▶ result   │───▶│  • Preview  │
│  • Webcam    │    │                                           │    │  • Reveal   │
│  • Demo      │    │  pencil_sketch   color_pencil             │    │  • Steps    │
└──────────────┘    │  cartoonify      ink_wash                 │    │  • PNG file │
       ▲            └───────────────────────────────────────────┘    └─────────────┘
       │                         ▲   │ timed() → ms / FPS                   │
       └──────────── app.py (Streamlit UI: sliders & tabs) ◀───────────────┘
```

**Design principles**
- **Separation of concerns:** `engine.py` has zero UI code, so it is reusable in scripts and notebooks.
- **Reuse:** Color Pencil and Ink & Wash build on `pencil_sketch` / `cartoonify`.
- **Transparency:** each pipeline can export its intermediate matrices (`steps` dict) for the inspector tab.
- **Performance:** down-scaling with `INTER_AREA`, and half-resolution bilateral filtering for large images.

## File Structure

```
chromasketch/
├── app.py                # Streamlit web interface
├── engine.py             # Image-processing algorithms (OpenCV + NumPy)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation & Usage

```bash
# 1. Clone
git clone https://github.com/<your-username>/chromasketch.git
cd chromasketch

# 2. (Optional) virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

### Use the engine without the UI

```python
import cv2, engine as E

img = cv2.imread("photo.jpg")
cv2.imwrite("pencil.png",  E.pencil_sketch(img, ksize=21, grain=0.25))
cv2.imwrite("colour.png",  E.color_pencil(img, sat=1.5))
cv2.imwrite("cartoon.png", E.cartoonify(img, iterations=5))
cv2.imwrite("inkwash.png", E.ink_wash(img))
```

## Parameters

| Parameter | Affects | Effect |
|---|---|---|
| Gaussian kernel | Pencil, Color Pencil | Larger → softer, bolder strokes |
| Paper grain | Pencil, Color Pencil | Paper texture strength |
| Saturation boost | Color Pencil | Colour vividness (default 1.5×) |
| Stroke contrast (γ) | Color Pencil | Darker / lighter pencil lines |
| Bilateral iterations | Cartoon | More → flatter, painted look |
| Edge block size | Cartoon | Larger → thicker, coarser outlines |
| Edge sensitivity (C) | Cartoon | Larger → fewer outlines |
| Poster colour bands | Cartoon / Ink & Wash | Limits number of colour levels |

## Performance

Measured on an 800×600 image (CPU only; your numbers will vary):

| Style | Time |
|---|---|
| Pencil Sketch | ~45 ms |
| Color Pencil | ~110 ms |
| Cartoonifier | ~65 ms |
| Ink & Wash | ~50 ms |

## Tech Stack

Python · OpenCV · NumPy · Pillow · Streamlit


## Future Work

- Live webcam video stream (streamlit-webrtc)
- Watercolour style via `cv2.stylization` / mean-shift filtering
- Batch processing and video-file stylization
- Face-aware line weighting with Haar cascades

## License

Released under the MIT License.
