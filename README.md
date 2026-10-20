<div align="center">

# ChromaSketch

### Artistic Cartoonifier and Color Sketch Generator

**Non-photorealistic rendering using classical computer vision. No neural networks. No GPU. Just OpenCV.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Array%20Math-013243?logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Hardware](https://img.shields.io/badge/Runs%20on-CPU%20Only-2ea44f)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Highlights](#key-highlights)
3. [The Three Styles](#the-three-styles)
4. [Methodology](#methodology)
5. [System Architecture](#system-architecture)
6. [Project Structure](#project-structure)
7. [Installation and Usage](#installation-and-usage)
8. [Using the Engine as a Library](#using-the-engine-as-a-library)
9. [Controls Reference](#controls-reference)
10. [Why Classical Computer Vision](#why-classical-computer-vision)
11. [Limitations and Future Work](#limitations-and-future-work)
12. [Tech Stack](#tech-stack)
13. [License](#license)

---

## Overview

**ChromaSketch** converts an ordinary photograph into three kinds of artwork: a graphite **pencil sketch**, a vibrant **color pencil drawing**, and a flat-shaded **comic cartoon**.

Most modern stylization tools rely on Generative Adversarial Networks or Neural Style Transfer. These need large trained models and GPU hardware. ChromaSketch shows that convincing artistic effects can be produced with a handful of well-chosen image-processing operators: spatial filtering, color-space manipulation and pixel-matrix arithmetic. Every output pixel can be traced back to a formula, which makes the whole pipeline transparent, explainable and fast.

## Key Highlights

| Highlight | Description |
|---|---|
| **CPU-only, real-time** | Runs in milliseconds on a normal laptop with no GPU and no model download. |
| **Novel color pencil engine** | Fuses a monochrome structural sketch mask with an HSV saturation-boosted copy of the photo. |
| **Edge-preserving smoothing** | Repeated bilateral filtering flattens skin, sky and walls while keeping object boundaries sharp. |
| **Clean separation of concerns** | Image processing lives in `engine.py`; the interface lives in `app.py`. |
| **Interactive web UI** | Upload, choose a style, tune parameters live, compare, and download. |
| **Fully explainable** | No black box. Each step is a simple, documented operation. |

## The Three Styles

| Style | Color Space | Primary Operator | Output Characteristics |
|---|---|---|---|
| **Pencil Sketch** | Grayscale | Color dodge (`cv2.divide`) | Monochrome graphite drawing |
| **Color Pencil** | BGR + HSV | Saturation boost + matrix multiplication | Textured, hand-colored lead drawing |
| **Cartoon** | BGR + Grayscale | Bilateral filter (5 iterations) + adaptive threshold | Flat, graphic comic-book art |

## Methodology

### 1. Pencil Sketch Engine

A pencil drawing is white paper with dark marks only where brightness changes quickly. The pipeline isolates exactly those regions.

1. **Grayscale conversion.** Convert BGR to a single channel, `I_gray`.
2. **Negative inversion.** `I_inv = 255 - I_gray`.
3. **Gaussian blur.** Apply a 21 x 21 kernel to `I_inv` to obtain `I_blur`.
4. **Color dodge blending.**

```
Sketch = I_gray / (255 - I_blur) * 255
```

In flat regions the pixel closely matches its blurred neighbourhood, so the ratio is about 1 and the output is white paper. Across edges the ratio drops below 1 and a dark stroke appears.

### 2. Color Pencil Engine

1. **HSV transformation.** Convert the photo to Hue, Saturation, Value. Saturation is an independent channel, so colors can be intensified without shifting hue or brightness.
2. **Saturation amplification.** Multiply S by 1.5 and clip to the valid range.
3. **Re-conversion and normalization.** Convert back to BGR and scale to the range 0.0 to 1.0.
4. **Mask compositing.** Multiply the normalized color matrix by the 3-channel pencil sketch. White paper areas let color pass through unchanged; pencil lines darken it.

### 3. Comic Cartoon Engine

The cartoon is built from two independent branches that are merged at the end.

**Branch A: outlines**
- Convert to grayscale and apply a 5 x 5 median blur to remove speckle noise while keeping edges sharp.
- Apply adaptive thresholding, which compares each pixel with the mean of its own neighbourhood. This produces clean black outlines even under uneven lighting.

**Branch B: flat colors**
- Apply a bilateral filter five times in sequence. It blurs similar-colored neighbours but refuses to blur across strong edges, so surfaces flatten while boundaries stay crisp.

**Merge**
- `cv2.bitwise_and(color, color, mask=edges)` uses the outline image as a stencil, stamping black lines over the flat colors.

## System Architecture

```
+----------------+      +-----------------------+      +--------------------------+
|  User uploads  | ---> |        app.py         | ---> |        engine.py         |
|  a photo       |      |  Streamlit interface  |      |  Image-processing core   |
+----------------+      |  - style selector     |      |                          |
                        |  - parameter sliders  |      |  resize_image()          |
                        |  - result display     |      |  pencil_sketch()         |
                        |  - download button    |      |  color_pencil()          |
                        +-----------------------+      |  cartoonify()            |
                                   ^                   +------------+-------------+
                                   |                                |
                                   +------- styled image -----------+
```

**Data flow**

```
Photo (RGB) -> convert to BGR -> resize -> chosen style function -> result -> display + PNG download
```

**Design principles**
- `engine.py` contains no UI code, so it can be reused in scripts, notebooks or other front ends.
- `app.py` contains no image-processing logic; it only collects inputs and displays outputs.
- Color Pencil reuses `pencil_sketch`, avoiding duplicated code.

## Project Structure

```
chromasketch_simple/
|-- app.py              # Streamlit web interface
|-- engine.py           # Image-processing algorithms (OpenCV + NumPy)
|-- requirements.txt    # Python dependencies
|-- EXPLANATION.txt     # Plain-language walkthrough of the code
`-- README.md           # Project documentation
```

| File | Responsibility |
|---|---|
| `engine.py` | Resizing and the three style pipelines. Pure functions that take an image and return an image. |
| `app.py` | Upload, style selection, sliders, side-by-side display and download. |
| `EXPLANATION.txt` | Beginner-friendly, step-by-step explanation of every function. |

## Installation and Usage

**Prerequisites:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/shindevaish/ChromaSketch.git
cd ChromaSketch 

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a photo, and pick a style.

## Using the Engine as a Library

The engine works on its own, with no UI required.

```python
import cv2
import engine

img = cv2.imread("photo.jpg")
img = engine.resize_image(img, max_size=800)

cv2.imwrite("pencil.png",  engine.pencil_sketch(img, blur_size=21))
cv2.imwrite("colour.png",  engine.color_pencil(img, saturation=1.5))
cv2.imwrite("cartoon.png", engine.cartoonify(img, iterations=5))
```

## Controls Reference

| Control | Applies To | Effect |
|---|---|---|
| Blur size | Pencil Sketch, Color Pencil | Larger values give softer, bolder shading and thicker strokes. |
| Color boost | Color Pencil | Saturation multiplier. Higher values give more vivid color. |
| Smoothness (filter passes) | Cartoon | More bilateral passes give flatter, more painted regions. |

## Why Classical Computer Vision

| Aspect | Classical (this project) | Deep learning approaches |
|---|---|---|
| Hardware | Any CPU | Usually a GPU |
| Setup | Four small Python packages | Large trained model files |
| Speed | Milliseconds per image | Slower without a GPU |
| Explainability | Every pixel traces to a formula | Largely a black box |
| Real-time tuning | Instant slider feedback | Often fixed after training |
| Trade-off | No semantic understanding of objects | Can learn rich artistic styles |

## Tech Stack

| Component | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Filtering, color conversion, thresholding, blending |
| NumPy | Pixel-matrix arithmetic |
| Pillow | Reading uploaded image files |
| Streamlit | Interactive web interface |