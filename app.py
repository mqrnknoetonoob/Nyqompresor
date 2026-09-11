"""
app.py
------
Streamlit UI: upload an image, compress with your own method AND OpenCV,
compare side by side with metrics.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image

from core import pipeline, opencv_baseline, metrics, test_patterns

st.set_page_config(page_title="Loss-Optimised Image Compressor", layout="wide")
st.title("Loss-Optimised 2D Image Compressor")
st.caption("Custom from-scratch pipeline vs OpenCV baseline — BUET CSE220 project")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")

source = st.sidebar.radio("Image source", ["Upload", "Synthetic test pattern"])

if source == "Upload":
    uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    image = None
    if uploaded is not None:
        image = np.array(Image.open(uploaded).convert("RGB"))
else:
    pattern_name = st.sidebar.selectbox("Pattern", ["Checkerboard", "Sine grating", "Radial zone plate"])
    size = st.sidebar.slider("Size (px)", 128, 512, 256, step=64)
    if pattern_name == "Checkerboard":
        square = st.sidebar.slider("Square size", 1, 16, 3)
        gray = test_patterns.checkerboard(size=size, square=square)
    elif pattern_name == "Sine grating":
        freq = st.sidebar.slider("Max frequency", 0.05, 1.0, 0.4)
        gray = test_patterns.sine_grating(size=size, frequency=freq)
    else:
        gray = test_patterns.radial_zone_plate(size=size)
    image = np.stack([gray, gray, gray], axis=-1)  # fake RGB for consistent pipeline

st.sidebar.divider()
factor = st.sidebar.slider("Downsampling factor", 2, 16, 4)
apply_filter = st.sidebar.checkbox("Apply anti-aliasing filter (your method)", value=True)
kernel_type = st.sidebar.selectbox("Filter kernel", ["gaussian", "box"])
kernel_size = st.sidebar.slider("Kernel size", 3, 15, 5, step=2)
recon_mode = st.sidebar.selectbox("Reconstruction mode (your method)", ["linear", "zoh"])

st.sidebar.divider()
cv_down = st.sidebar.selectbox("OpenCV downsample interp", ["area", "linear", "nearest"])
cv_up = st.sidebar.selectbox("OpenCV upsample interp", ["linear", "nearest", "cubic"])

# ---------------- Main ----------------
if image is None:
    st.info("Upload an image or pick a synthetic test pattern from the sidebar to begin.")
    st.stop()

if image.shape[0] > 1000 or image.shape[1] > 1000:
    st.warning("Large image detected — downscaling to 800px max side for speed in this demo.")
    scale = 800 / max(image.shape[0], image.shape[1])
    new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
    image = np.array(Image.fromarray(image).resize((new_w, new_h)))

with st.spinner("Running your custom pipeline..."):
    own_result = pipeline.run_pipeline(
        image, factor=factor, recon_mode=recon_mode,
        kernel_size=kernel_size, kernel_type=kernel_type, apply_filter=apply_filter,
    )

with st.spinner("Running OpenCV baseline..."):
    cv_result = opencv_baseline.opencv_round_trip(
        image, factor=factor, down_interp=cv_down, up_interp=cv_up,
    )

own_metrics = metrics.summarize(image, own_result["reconstructed"], own_result["downsampled"].shape)
cv_metrics = metrics.summarize(image, cv_result["reconstructed"], cv_result["downsampled"].shape)

# ---------------- Display ----------------
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Your Method")
    st.image(own_result["reconstructed"], use_container_width=True)
with col3:
    st.subheader("OpenCV Baseline")
    st.image(cv_result["reconstructed"], use_container_width=True)

st.divider()
st.subheader("Comparison Report")

report_col1, report_col2 = st.columns(2)
with report_col1:
    st.markdown("**Your Method**")
    st.metric("PSNR (dB)", f"{own_metrics['psnr_db']:.2f}")
    st.metric("MSE", f"{own_metrics['mse']:.2f}")
    st.metric("Compression ratio", f"{own_metrics['compression_ratio']:.2f}x")
with report_col2:
    st.markdown("**OpenCV Baseline**")
    st.metric("PSNR (dB)", f"{cv_metrics['psnr_db']:.2f}")
    st.metric("MSE", f"{cv_metrics['mse']:.2f}")
    st.metric("Compression ratio", f"{cv_metrics['compression_ratio']:.2f}x")

winner = "Your method" if own_metrics["psnr_db"] > cv_metrics["psnr_db"] else "OpenCV baseline"
st.success(f"Higher PSNR at this setting: **{winner}**")

with st.expander("Downsampled (stored) images — what's actually being kept"):
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.caption(f"Your method — shape {own_result['downsampled'].shape}")
        st.image(own_result["downsampled"], use_container_width=True)
    with dcol2:
        st.caption(f"OpenCV — shape {cv_result['downsampled'].shape}")
        st.image(cv_result["downsampled"], use_container_width=True)

st.caption(
    "Note: 'Compression ratio' here reflects pixel-count reduction, not final file "
    "byte size. Add entropy coding (RLE/Huffman) on the downsampled array for true "
    "byte-level compression figures."
)
