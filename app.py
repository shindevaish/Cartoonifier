import cv2, io
import numpy as np
import streamlit as st
from PIL import Image
import engine as E

st.set_page_config(page_title="ChromaSketch", page_icon="✏️", layout="wide")
st.markdown("""
<style>
.hero{padding:1.4rem 1.6rem;border-radius:18px;margin-bottom:1rem;
background:linear-gradient(120deg,#ff9a5a 0%,#ff5f8f 45%,#6a5cff 100%);color:#fff}
.hero h1{margin:0;font-size:2.2rem;letter-spacing:-.5px}
.hero p{margin:.2rem 0 0;opacity:.92}
.chip{display:inline-block;padding:.15rem .7rem;border-radius:99px;background:#6a5cff22;
color:#6a5cff;font-weight:600;font-size:.8rem;margin-right:.4rem}
</style>
<div class="hero"><h1>✏️ ChromaSketch</h1>
<p>Artistic Cartoonifier &amp; Color Sketch Generator · classical computer vision, CPU only, no neural nets</p></div>
""", unsafe_allow_html=True)


def demo_image():
    """Built-in demo scene so the app works with no upload."""
    img = np.zeros((420, 640, 3), np.uint8)
    for y in range(420):
        img[y, :] = (255 - y // 3, 190 - y // 5, 120 + y // 6)
    cv2.circle(img, (500, 90), 55, (60, 200, 255), -1)
    cv2.fillPoly(img, [np.array([[0, 420], [200, 170], [380, 420]])], (70, 110, 60))
    cv2.fillPoly(img, [np.array([[250, 420], [450, 220], [640, 420]])], (50, 150, 90))
    cv2.rectangle(img, (300, 300), (380, 380), (40, 60, 200), -1)
    cv2.fillPoly(img, [np.array([[290, 300], [340, 255], [390, 300]])], (30, 30, 130))
    return cv2.GaussianBlur(img, (3, 3), 0)


# ---------- sidebar ----------
with st.sidebar:
    st.header("1 · Input")
    src = st.radio("Source", ["Upload", "Webcam", "Demo scene"], horizontal=True)
    bgr = None
    if src == "Upload":
        f = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
        if f: bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
    elif src == "Webcam":
        f = st.camera_input("Snap a photo")
        if f: bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        bgr = demo_image()
    max_side = st.slider("Max resolution (px)", 400, 2000, 1000, 100)

    st.header("2 · Style")
    mode = st.selectbox("Engine", ["✏️ Pencil Sketch", "🎨 Color Pencil", "💥 Cartoonifier",
                                   "🖋️ Ink & Wash (bonus)", "🧩 All styles"])
    st.header("3 · Parameters")
    ksize = st.slider("Gaussian kernel (pencil softness)", 3, 61, 21, 2)
    grain = st.slider("Paper grain", 0.0, 1.0, 0.25, 0.05)
    sat = st.slider("Saturation boost (S ×)", 1.0, 3.0, 1.5, 0.1)
    gamma = st.slider("Stroke contrast (γ)", 0.5, 2.5, 1.0, 0.1)
    iters = st.slider("Bilateral iterations", 1, 10, 5)
    block = st.slider("Edge block size", 3, 25, 9, 2)
    cval = st.slider("Edge sensitivity (C)", 0, 10, 2)
    poster = st.slider("Poster color bands (0 = off)", 0, 12, 0)

if bgr is None:
    st.info("👈 Upload a photo, snap one with your webcam, or pick the demo scene.")
    st.stop()

bgr = E.resize_max(bgr, max_side)
steps = {}


def run(name):
    if name.startswith("✏️"): return E.timed(E.pencil_sketch, bgr, ksize, grain, steps=steps)
    if name.startswith("🎨"): return E.timed(E.color_pencil, bgr, sat, ksize, gamma, grain, steps=steps)
    if name.startswith("💥"): return E.timed(E.cartoonify, bgr, iters, block, cval, 75, poster, steps=steps)
    return E.timed(E.ink_wash, bgr, 0.6, iterations=3, posterize=poster or 6)


def rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


if mode.startswith("🧩"):
    names = ["✏️ Pencil Sketch", "🎨 Color Pencil", "💥 Cartoonifier", "🖋️ Ink & Wash (bonus)"]
    cols = st.columns(3)
    cols[0].image(rgb(bgr), caption="Original", use_container_width=True)
    for i, n in enumerate(names):
        out, ms = run(n)
        cols[(i + 1) % 3].image(rgb(out), caption=f"{n} · {ms:.0f} ms", use_container_width=True)
    st.stop()

out, ms = run(mode)
m1, m2, m3 = st.columns(3)
m1.metric("Render time", f"{ms:.0f} ms")
m2.metric("FPS equivalent", f"{1000 / max(ms, 1):.1f}")
m3.metric("Resolution", f"{bgr.shape[1]}×{bgr.shape[0]}")

t1, t2, t3, t4 = st.tabs(["Result", "Before / After reveal", "Pipeline inspector", "Theory"])
with t1:
    a, b = st.columns(2)
    a.image(rgb(bgr), caption="Original", use_container_width=True)
    b.image(rgb(out), caption=mode, use_container_width=True)
    buf = io.BytesIO(); Image.fromarray(rgb(out)).save(buf, "PNG")
    st.download_button("⬇️ Download PNG", buf.getvalue(), "chromasketch.png", "image/png")
with t2:
    x = st.slider("Reveal ←→", 0, 100, 50)
    o = rgb(out).copy(); w = int(o.shape[1] * x / 100)
    o[:, :w] = rgb(bgr)[:, :w]
    if 0 < w < o.shape[1]: o[:, max(w - 1, 0):w + 1] = (255, 255, 255)
    st.image(o, use_container_width=True)
with t3:
    st.caption("Every intermediate matrix of the selected pipeline.")
    cols = st.columns(len(steps))
    for c, (k, v) in zip(cols, steps.items()):
        c.image(rgb(v), caption=k, use_container_width=True)
with t4:
    st.latex(r"\text{Sketch}=\frac{I_{gray}}{255-I_{blur}}\times 255")
    st.markdown("""
- **Pencil:** gray → invert → Gaussian blur → colour-dodge divide.
- **Color pencil:** HSV saturation ×k → back to BGR → normalise → multiply by sketch mask.
- **Cartoon:** median blur + adaptive threshold for edges, N× bilateral filter for flat colour, bitwise-AND.
- **Bonus:** paper grain, poster bands, and *Ink & Wash* (cartoon wash under pencil lines).""")
