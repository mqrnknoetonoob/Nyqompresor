"""
app.py
------
Streamlit UI: upload an image, compress with the Nyquist-Shannon custom
method, OpenCV, and YCbCr chroma subsampling.  Compare all three side by
side with both signal-quality metrics (PSNR, MSE) and real byte-level
compression metrics (zlib-compressed stored size vs original).

Run with: streamlit run app.py
"""

import math
import streamlit as st
import numpy as np
from PIL import Image

from core import pipeline, opencv_baseline, metrics, test_patterns, color as color_mod


def ratio_to_factor(ratio: float) -> int:
    """
    Convert a user-supplied compression ratio to the nearest valid integer
    downsampling factor.  CR = factor², so factor = sqrt(CR), rounded.
    Clamped to a minimum of 1 (no downsampling).
    """
    return max(1, round(math.sqrt(max(1.0, ratio))))


def fmt_bytes(n: int) -> str:
    """Human-readable byte size (B / KB / MB)."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 ** 2):.2f} MB"


def is_grayscale(image: np.ndarray) -> bool:
    """
    True when all three RGB channels are identical (i.e. the image is
    effectively grayscale, even if stored as RGB).  Synthetic test
    patterns are constructed this way.
    """
    return (
        np.array_equal(image[:, :, 0], image[:, :, 1]) and
        np.array_equal(image[:, :, 1], image[:, :, 2])
    )


st.set_page_config(page_title="Loss-Optimised Image Compressor", layout="wide")
st.title("Loss-Optimised 2D Image Compressor")
st.caption("Comparison between different methods — BUET CSE220 project")

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

# --- Nyquist-Shannon / custom pipeline ---
st.sidebar.divider()
st.sidebar.subheader("Nyquist-Shannon Pipeline")
factor = st.sidebar.slider("Downsampling factor", 2, 16, 4)
apply_filter = st.sidebar.checkbox("Apply anti-aliasing filter", value=True)
kernel_type = st.sidebar.selectbox("Filter kernel", ["gaussian", "box"])
kernel_size = st.sidebar.slider("Kernel size", 3, 15, 5, step=2)
recon_mode = st.sidebar.selectbox("Reconstruction mode", ["linear", "zoh"])

# --- OpenCV baseline ---
st.sidebar.divider()
st.sidebar.subheader("OpenCV Baseline")
cv_down = st.sidebar.selectbox("OpenCV downsample interp", ["area", "linear", "nearest"])
cv_up   = st.sidebar.selectbox("OpenCV upsample interp",   ["linear", "nearest", "cubic"])

# --- Chroma subsampling ---
st.sidebar.divider()
st.sidebar.subheader("Chroma Subsampling (YCbCr)")

chroma_input_mode = st.sidebar.radio(
    "Set chroma factors by",
    ["Integer factor", "Compression ratio"],
    help=(
        "Integer factor: directly choose how many pixels to skip.\n"
        "Compression ratio: enter any real number and the nearest valid "
        "integer factor is derived automatically (CR ≈ factor²)."
    ),
)

if chroma_input_mode == "Integer factor":
    luma_factor = int(st.sidebar.number_input(
        "Luma (Y) factor", min_value=1, max_value=32, value=1, step=1,
        help="1 = no luma downsampling (recommended — eye is very sensitive to brightness).",
    ))
    chroma_factor = int(st.sidebar.number_input(
        "Chroma (Cb, Cr) factor", min_value=1, max_value=32, value=4, step=1,
        help="Downsample colour channels more aggressively. 4 is a typical real-world choice.",
    ))
else:
    luma_cr = st.sidebar.number_input(
        "Luma compression ratio", min_value=1.0, value=1.0, step=0.5,
        help="Any real number ≥ 1. Nearest integer factor will be derived.",
    )
    chroma_cr = st.sidebar.number_input(
        "Chroma compression ratio", min_value=1.0, value=16.0, step=0.5,
        help="Any real number ≥ 1. Nearest integer factor will be derived.",
    )
    luma_factor   = ratio_to_factor(luma_cr)
    chroma_factor = ratio_to_factor(chroma_cr)

st.sidebar.caption(
    f"Effective factors → Y: **{luma_factor}×**, Cb/Cr: **{chroma_factor}×**  \n"
    f"Approx. per-plane CR → Y: {luma_factor ** 2}×, Cb/Cr: {chroma_factor ** 2}×"
)

# ---------------- Main ----------------
if image is None:
    st.info("Upload an image or pick a synthetic test pattern from the sidebar to begin.")
    st.stop()

if image.shape[0] > 1000 or image.shape[1] > 1000:
    st.warning("Large image detected — downscaling to 800px max side for speed in this demo.")
    scale = 800 / max(image.shape[0], image.shape[1])
    new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
    image = np.array(Image.fromarray(image).resize((new_w, new_h)))

# ── Grayscale / synthetic pattern warning for chroma ──────────────────────────
grayscale_input = is_grayscale(image)
if grayscale_input:
    st.info(
        "**Chroma subsampling note — grayscale input detected.**  \n"
        "Synthetic test patterns have no colour: when converted to YCbCr the Cb and Cr "
        "channels are flat (constant 128 — neutral grey). Subsampling a constant channel "
        "by any factor and reconstructing it introduces **zero error**, so MSE = 0 and "
        "PSNR = ∞ for chroma subsampling on this image. This is mathematically correct, "
        "not a bug.  \n"
        "**Upload a real colour photo** to see meaningful chroma-subsampling distortion "
        "and a meaningful PSNR comparison."
    )
# ──────────────────────────────────────────────────────────────────────────────

with st.spinner("Running Nyquist-Shannon custom pipeline..."):
    own_result = pipeline.run_pipeline(
        image, factor=factor, recon_mode=recon_mode,
        kernel_size=kernel_size, kernel_type=kernel_type, apply_filter=apply_filter,
    )

with st.spinner("Running OpenCV baseline..."):
    cv_result = opencv_baseline.opencv_round_trip(
        image, factor=factor, down_interp=cv_down, up_interp=cv_up,
    )

with st.spinner("Running YCbCr chroma subsampling pipeline..."):
    chroma_result = color_mod.chroma_subsample_pipeline(
        image,
        luma_factor=luma_factor,
        chroma_factor=chroma_factor,
        recon_mode=recon_mode,
        kernel_size=kernel_size,
    )

# --- Compute metrics (full: pixel-count + byte-level) ---
own_metrics    = metrics.full_summary(image, own_result["reconstructed"],
                                      own_result["downsampled"])
cv_metrics     = metrics.full_summary(image, cv_result["reconstructed"],
                                      cv_result["downsampled"])
chroma_metrics = metrics.chroma_full_summary(
    image,
    chroma_result["reconstructed"],
    chroma_result["downsampled_planes"],
    luma_factor, chroma_factor,
)

# Determine whether chroma PSNR is meaningful for display
chroma_psnr_inf = chroma_metrics["psnr_db"] == float("inf")

# ---------------- Display ----------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.subheader("Original")
    st.image(image, use_container_width=True)
with col2:
    st.subheader("Nyquist-Shannon Method")
    st.image(own_result["reconstructed"], use_container_width=True)
with col3:
    st.subheader("OpenCV Baseline")
    st.image(cv_result["reconstructed"], use_container_width=True)
with col4:
    st.subheader("Chroma Subsampling")
    st.image(chroma_result["reconstructed"], use_container_width=True)

st.divider()
st.subheader("Comparison between different methods")


def fmt_psnr(v: float) -> str:
    return "∞  (lossless)" if v == float("inf") else f"{v:.2f}"


def render_metrics_col(m: dict, label: str, extra_caption: str = ""):
    st.markdown(f"**{label}**")
    st.metric("PSNR (dB)",
              fmt_psnr(m["psnr_db"]),
              help=">40 dB excellent · 30–40 dB good · <20 dB poor")
    st.metric("MSE", f"{m['mse']:.2f}",
              help="Mean Squared Error — lower is better")
    st.metric("Pixel-count CR",
              f"{m['compression_ratio']:.2f}×",
              help="Ratio of original to stored pixel count (spatial only)")
    st.divider()
    st.markdown("**Byte-level compression** *(zlib / deflate)*")
    st.metric("Original size",  fmt_bytes(m["original_bytes"]))
    st.metric("Stored size",    fmt_bytes(m["stored_bytes"]),
              delta=f"−{fmt_bytes(m['original_bytes'] - m['stored_bytes'])} saved",
              delta_color="inverse")
    st.metric("Byte CR",
              f"{m['byte_compression_ratio']:.2f}×",
              help="Compressed original ÷ compressed stored data (honest file-size saving)")
    if extra_caption:
        st.caption(extra_caption)


report_col1, report_col2, report_col3 = st.columns(3)
with report_col1:
    render_metrics_col(own_metrics, "Nyquist-Shannon Method")
with report_col2:
    render_metrics_col(cv_metrics, "OpenCV Baseline")
with report_col3:
    render_metrics_col(
        chroma_metrics, "Chroma Subsampling",
        extra_caption=f"Y: {luma_factor}×  |  Cb/Cr: {chroma_factor}×",
    )

# ── Winner banners ─────────────────────────────────────────────────────────────

# PSNR winner — exclude ∞ from chroma when input is grayscale (not a fair comparison)
if chroma_psnr_inf and grayscale_input:
    psnr_scores = {
        "Nyquist-Shannon Method": own_metrics["psnr_db"],
        "OpenCV Baseline":        cv_metrics["psnr_db"],
    }
    psnr_winner = max(psnr_scores, key=psnr_scores.get)
    st.success(
        f"Highest PSNR (colour images only): **{psnr_winner}**  \n"
        "*(Chroma subsampling shows ∞ PSNR on grayscale — see note above)*"
    )
else:
    psnr_scores = {
        "Nyquist-Shannon Method": own_metrics["psnr_db"],
        "OpenCV Baseline":        cv_metrics["psnr_db"],
        "Chroma Subsampling":     chroma_metrics["psnr_db"],
    }
    psnr_winner = max(psnr_scores, key=psnr_scores.get)
    st.success(f"Highest PSNR at this setting: **{psnr_winner}**")

# Byte CR winner — always meaningful
byte_scores = {
    "Nyquist-Shannon Method": own_metrics["byte_compression_ratio"],
    "OpenCV Baseline":        cv_metrics["byte_compression_ratio"],
    "Chroma Subsampling":     chroma_metrics["byte_compression_ratio"],
}
best_byte = max(byte_scores, key=byte_scores.get)
st.info(
    f"Best byte compression ratio: **{best_byte}** "
    f"({byte_scores[best_byte]:.2f}× vs original compressed size)  \n"
    "*(Chroma subsampling optimises for compression ratio + perceptual quality, "
    "not raw PSNR — see explanation below)*"
)
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("ℹ️  Why chroma subsampling may not win on PSNR — but that's the point"):
    st.markdown(
        """
        **What chroma subsampling optimises for:**
        - It exploits the **Human Visual System (HVS)**: the eye has ~4× lower resolution
          sensitivity to colour (chrominance) than to brightness (luminance).
        - By keeping the Y (luma) channel intact and only discarding Cb/Cr detail,
          the image looks nearly identical to humans even though the data is smaller.
        - This is exactly how **JPEG, H.264 video, and almost every real codec** work.

        **Why PSNR/MSE penalise it anyway:**
        - PSNR and MSE treat all channels equally — a 1-unit error in Cb is counted
          the same as a 1-unit error in Y, even though the eye barely notices the former.
        - So chroma subsampling will usually score **lower PSNR than the Nyquist-Shannon
          method** on colour images, while **looking equally good or better** visually.
        - The right metric to compare is **Byte CR** (or a perceptual metric like SSIM),
          not raw PSNR.

        **On grayscale / synthetic test patterns:**
        - The Cb and Cr channels are flat (constant 128 — neutral grey with no colour).
        - Downsampling a constant plane introduces zero error → MSE = 0, PSNR = ∞.
        - This is mathematically correct. Upload a colour photo to see real distortion.
        """
    )

with st.expander("Downsampled (stored) images — what's actually being kept"):
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.caption(f"Nyquist-Shannon method — shape {own_result['downsampled'].shape}")
        st.image(own_result["downsampled"], use_container_width=True)
    with dcol2:
        st.caption(f"OpenCV — shape {cv_result['downsampled'].shape}")
        st.image(cv_result["downsampled"], use_container_width=True)
    st.caption(
        "Chroma subsampling stores three separate YCbCr planes at different resolutions "
        "internally — there is no single combined 'downsampled' image to display here."
    )

st.caption(
    "Byte CR uses zlib (deflate level 6) applied to the raw uint8 pixel bytes of the "
    "stored array. Both the original and the stored data are compressed before taking "
    "the ratio, so the number reflects net savings from spatial decimation alone, not "
    "from the entropy coder. A real file format (PNG, JPEG, HEIF) would add further "
    "prediction and entropy coding on top of this."
)
