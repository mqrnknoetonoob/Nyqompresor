# Loss-Optimised 2D Image Compressor (BUET CSE220)

Custom Nyquist-aware compressor (filter → downsample → interpolate),
compared against OpenCV's baseline resize.

## Project structure

```
core/
  filters.py          Low-pass filter (box/Gaussian) + manual 2D convolution
  downsample.py        Decimation (every Nth pixel)
  interpolate.py        Reconstruction: ZOH (nearest) and bilinear, from scratch
  metrics.py             MSE, PSNR, compression ratio
  pipeline.py             Your full encode/decode chain (uses the above)
  opencv_baseline.py    The "black box" comparison method (cv2.resize only)
  color.py                Optional stretch goal: YCbCr chroma subsampling
  test_patterns.py     Synthetic images (checkerboard, sine grating, zone plate)
                          that clearly show aliasing — use these for your
                          filter-on vs filter-off report figure
app.py                   Streamlit UI
test_pipeline.py       Smoke test — run this first to confirm everything works
```

## Setup

```bash
pip install numpy opencv-python-headless pillow streamlit
```

## Run

```bash
# Sanity check the pipeline (no UI)
python3 test_pipeline.py

# Launch the UI
streamlit run app.py
```

## How the pieces map to theory (for your report)

| Concept | File | Function |
|---|---|---|
| Anti-aliasing (Nyquist) | filters.py | low_pass_filter, convolve2d |
| Decimation | downsample.py | downsample |
| ZOH reconstruction | interpolate.py | upsample_nearest |
| Linear/bilinear reconstruction | interpolate.py | upsample_bilinear |
| Quality evaluation | metrics.py | mse, psnr |
| Full round trip | pipeline.py | run_pipeline |
| OpenCV baseline | opencv_baseline.py | opencv_round_trip |
| Chroma subsampling (stretch) | color.py | chroma_subsample_pipeline |

## Key experiments to run for your report

1. **Aliasing demo (the core Nyquist proof):** use `test_patterns.checkerboard()`
   or `radial_zone_plate()`, run `pipeline.run_pipeline(..., apply_filter=True)`
   vs `apply_filter=False`, compare visually and via PSNR. Filtered should
   show noticeably fewer moire artifacts and better PSNR at high downsampling
   factors.

2. **ZOH vs linear reconstruction:** same downsampled data, `recon_mode="zoh"`
   vs `"linear"`. Linear should look smoother and score higher PSNR (usually).

3. **Rate-distortion curve:** loop `factor` from 2 to 16, plot PSNR (y-axis)
   vs compression ratio (x-axis). Expect PSNR to drop as ratio increases —
   this trade-off curve is a classic compression report figure.

4. **Your method vs OpenCV:** run both pipelines at the same factor on the
   same image, compare PSNR/MSE. Try `cv_down="linear"` (naive, no proper
   anti-aliasing) vs `cv_down="area"` (OpenCV's actual anti-aliased shrink)
   to see how much that choice matters — a genuinely interesting result.

## Notes / honest limitations to mention in your report

- "Compression ratio" here is pixel-count reduction, not final file byte
  size. For true byte-level savings, add a simple entropy coder (RLE or
  Huffman) on top of the downsampled array — noted as a possible extension.
- The from-scratch convolution/interpolation is implemented in Python/numpy
  and will be slower than OpenCV's compiled backend — worth reporting
  honestly as a runtime comparison, not hiding it.
- `should_compress()` in `pipeline.py` is a stub for a planned future
  extension (auto-detecting sensitive documents to gate compression) —
  not implemented in this version, deliberately left as an extension point.
