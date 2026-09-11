"""Smoke test: run the pipeline on synthetic + real-shaped data, check no crashes
and metrics look sane. Not a formal unit test suite -- just a sanity check."""

import numpy as np
from core import pipeline, opencv_baseline, metrics, test_patterns, color

print("=== Test 1: grayscale checkerboard, filter ON vs OFF ===")
img = test_patterns.checkerboard(size=256, square=3)
img3 = img[:, :, np.newaxis]  # (H, W, 1) to exercise the color-capable path

for apply_filter in [True, False]:
    result = pipeline.run_pipeline(img3, factor=4, recon_mode="linear", apply_filter=apply_filter)
    m = metrics.summarize(img3, result["reconstructed"], result["downsampled"].shape)
    print(f"  filter={apply_filter}: MSE={m['mse']:.2f}  PSNR={m['psnr_db']:.2f}dB  ratio={m['compression_ratio']:.2f}x")

print("\n=== Test 2: ZOH vs linear reconstruction ===")
for mode in ["zoh", "linear"]:
    result = pipeline.run_pipeline(img3, factor=4, recon_mode=mode, apply_filter=True)
    m = metrics.summarize(img3, result["reconstructed"], result["downsampled"].shape)
    print(f"  mode={mode}: MSE={m['mse']:.2f}  PSNR={m['psnr_db']:.2f}dB")

print("\n=== Test 3: RGB image (random, simulating a photo) ===")
rgb = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
result = pipeline.run_pipeline(rgb, factor=3, recon_mode="linear", apply_filter=True)
print(f"  original shape={rgb.shape} -> downsampled={result['downsampled'].shape} -> reconstructed={result['reconstructed'].shape}")
m = metrics.summarize(rgb, result["reconstructed"], result["downsampled"].shape)
print(f"  MSE={m['mse']:.2f}  PSNR={m['psnr_db']:.2f}dB  ratio={m['compression_ratio']:.2f}x")

print("\n=== Test 4: OpenCV baseline comparison ===")
cv_result = opencv_baseline.opencv_round_trip(rgb, factor=3, down_interp="area", up_interp="linear")
m_cv = metrics.summarize(rgb, cv_result["reconstructed"], cv_result["downsampled"].shape)
print(f"  OpenCV: MSE={m_cv['mse']:.2f}  PSNR={m_cv['psnr_db']:.2f}dB  ratio={m_cv['compression_ratio']:.2f}x")

print("\n=== Test 5: YCbCr chroma subsampling ===")
out = color.chroma_subsample_pipeline(rgb, luma_factor=1, chroma_factor=4)
print(f"  output shape={out.shape}, dtype={out.dtype}")
m_chroma = metrics.summarize(rgb, out, (rgb.shape[0]//1, rgb.shape[1]//1))
print(f"  MSE={m_chroma['mse']:.2f}  PSNR={m_chroma['psnr_db']:.2f}dB")

print("\nAll tests completed without crashing.")
