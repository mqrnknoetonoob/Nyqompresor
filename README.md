# Nyqompressor: Loss-Optimised 2D Image Compressor

An end-to-end Python framework implementing custom 2D discrete convolution, spatial decimation, and reconstruction pipelines from scratch to benchmark Nyquist–Shannon sampling theorem-driven image compression against industrial baselines (OpenCV).

---

## 1. Project Overview & Motivation

Digital images are two-dimensional discrete spatial signals $f(x, y)$. In standard image compression and resizing pipelines, naive downsampling often discards spatial samples indiscriminately. According to the **Nyquist–Shannon Sampling Theorem**, violating the sampling criterion introduces **aliasing**—manifesting visually as Moiré patterns, jagged edges, and severe high-frequency distortion.

**Nyqompressor** builds an honest, ground-up pipeline implementing:
- Anti-aliasing spatial low-pass filtering via direct 2D convolution.
- Signal decimation (subsampling).
- Reconstruction/interpolation (Zero-Order Hold vs. Bilinear Interpolation).
- Quantitative rate-distortion benchmarks ($\text{PSNR}$, $\text{MSE}$, Compression Ratio).
- Direct comparison with OpenCV (`cv2.resize`).
- Chroma subsampling ($YC_bC_r$) based on the Human Visual System (HVS).

---

## 2. Theoretical Pipeline Architecture

```
[ Input 2D Image f(x, y) ]
            │
            ▼
┌──────────────────────────────────────┐
│  1. Anti-Aliasing Low-Pass Filter    │  g(x, y) = f(x, y) * h(x, y)
│     (Gaussian / Box 2D Convolution)  │  Enforces Nyquist limit: f_max < f_s / (2M)
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  2. Decimation / Subsampling         │  s(x, y) = g(M·x, M·y)
│     (Spatial Dimension Reduction)    │  Pixel Count Reduction Ratio: M^2
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  3. Reconstruction / Interpolation   │  Rebuilds original grid f̂(x, y)
│     (Zero-Order Hold vs. Bilinear)   │  Low-pass reconstruction filter
└──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────┐
│  4. Quantitative Evaluation          │  MSE, PSNR (dB), Rate-Distortion curves,
│     & OpenCV Benchmarking            │  vs. cv2.INTER_AREA / cv2.INTER_LINEAR
└──────────────────────────────────────┘
```

---

## 3. Core Theoretical Stages

### Stage 1: Anti-Aliasing Filtering (Nyquist–Shannon Condition)
When an image is downsampled by an integer factor $M$, the new spatial sampling frequency becomes:
$$f_s' = \frac{f_s}{M}$$

To prevent spectral overlapping (aliasing), all spatial frequencies exceeding the new Nyquist limit must be attenuated before downsampling:
$$f_{\text{max}} < \frac{f_s'}{2} = \frac{f_s}{2M}$$

We implement spatial domain 2D discrete convolution from scratch:
$$g(x, y) = f(x, y) * h(x, y) = \sum_{m} \sum_{n} f(m, n) \cdot h(x - m, y - n)$$

- **Gaussian Kernel:** Smooth frequency roll-off without spatial ringings.
- **Box Kernel:** Uniform spatial moving average window.

### Stage 2: Decimation (Compression)
Decimation retains every $M$-th pixel along spatial coordinates:
$$s(x, y) = g(M \cdot x, M \cdot y)$$
The compression ratio ($\text{CR}$) based on pixel footprint is:
$$\text{CR} = \frac{N_{\text{original}}}{N_{\text{downsampled}}} = M^2$$

### Stage 3: Image Reconstruction (Interpolation / Upsampling)
Two distinct reconstruction filters are implemented from scratch:
1. **Zero-Order Hold (ZOH / Nearest-Neighbor):**
   $$\hat{f}(x, y) = s\left(\left\lfloor \frac{x}{M} \right\rfloor, \left\lfloor \frac{y}{M} \right\rfloor\right)$$
   Produces blocky, pixelated visual artifacts with discontinuous edges.
2. **First-Order Hold (Bilinear Interpolation):**
   Performs separable 1D linear interpolations along rows followed by columns:
   $$\hat{f}(x, y) \approx (1-\alpha)(1-\beta)s_{i,j} + \alpha(1-\beta)s_{i+1,j} + (1-\alpha)\beta s_{i,j+1} + \alpha\beta s_{i+1,j+1}$$
   Produces continuous, smooth transitions and significantly higher fidelity.

### Stage 4: Rate-Distortion & Quality Metrics
Reconstruction quality is evaluated quantitatively:
- **Mean Squared Error (MSE):**
  $$\text{MSE} = \frac{1}{H \times W} \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} [f(x, y) - \hat{f}(x, y)]^2$$
- **Peak Signal-to-Noise Ratio (PSNR):**
  $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \quad (\text{dB})$$

### Stage 5: Human Visual System (HVS) & Chroma Subsampling
The human eye possesses significantly higher visual acuity for brightness (Luminance $Y$) than color differences (Chrominance $C_b, C_r$). Converting RGB to $YC_bC_r$ allows heavy decimation on chroma components while preserving luma, yielding massive compression with minimal perceived degradation.

---

## 4. Codebase Architecture

```
Claude_Generated_Demo/
├── core/
│   ├── filters.py          # Vectorized 2D convolution & Gaussian/Box kernels
│   ├── downsample.py       # Spatial decimation and compression ratio calculators
│   ├── interpolate.py     # From-scratch ZOH and Bilinear interpolation engines
│   ├── metrics.py          # MSE, PSNR (dB), and statistical summaries
│   ├── pipeline.py         # Full encode-decode pipeline orchestrator
│   ├── opencv_baseline.py  # Industrial benchmark wrapper (cv2.resize)
│   ├── color.py            # YCbCr conversion and asymmetric chroma subsampling
│   └── test_patterns.py    # Synthetic signals (Checkerboard, Sine Grating, Zone Plate)
├── app.py                  # Interactive Streamlit UI with side-by-side comparison
├── test_pipeline.py        # Automated smoke test suite
├── README.md               # English project documentation
└── README_BN.md            # বাংলা প্রজেক্ট ডকুমেন্টেশন
```

---

## 5. How to Run

### Prerequisites
Ensure dependencies are installed:
```bash
py -m pip install numpy opencv-python pillow streamlit
```

### 1. Run Automated Console Tests
```bash
py test_pipeline.py
```

### 2. Launch Interactive Streamlit GUI
```bash
py -m streamlit run app.py
```
Access the application at `http://localhost:8501`.
