# GPU development and verification

- Keep the native CPU engine, GUI, and camera presets unchanged when working on the additive `brilliantisp_gpu` implementation. PyTorch imports must remain lazy.
- Use the separate `.venv-gpu` environment; do not install GPU dependencies into the existing `venv`.
- Run targeted GPU tests from the repository root, for example:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg .venv-gpu/bin/python -m pytest tests/gpu/test_demosaic_bilinear.py tests/gpu/test_demosaic_malvar.py -q --require-gpu --gpu-vendor nvidia`
- `--require-gpu` makes unavailable hardware a failure. Optional CPU-only runs and hardware skips are not qualification. AMD qualification requires a suitable ROCm PyTorch environment and `--gpu-vendor amd`; it has not been performed on this host.
- Preserve numerical gates in `tools/gpu_validation/contracts.py`. Normalize comparisons by the declared signal scale, never the observed peak or a headroom multiplier.
- The current numerical reference uses NumPy 1.26.4 and SciPy 1.17.1. SciPy `ndimage` convolution accumulates differently from `signal.correlate2d`: the latter uses four-tap float32 groups for the Malvar reference. The frozen grouping fixture is `tests/gpu/fixtures/malvar_signal_grouping.npz`.
- AE uses float64 sums divided by a device float64 count rather than `torch.mean`; reciprocal averaging can change a nearest-gain decision. Preserve `tests/gpu/fixtures/ae_rounding_cases.json`.
- Full-resolution reference captures belong under `tmp/gpu_validation`; keep them compressed and deduplicated because local disk headroom is limited.
- GPU native output uses physical `torch.uint8`/`torch.uint16` storage, including packed YCbCr, and previews use `torch.uint8`. Assert dtype before numerical comparisons; casting actual output to the expected dtype can hide contract failures.
- Native tone mapping requires an explicit GPU `color_model` (`bayer` or `rgb`) to distinguish BHW from HWC3. Keep mapped luminance in float64 through RGB recombination, including the fixed Q8.24 bridge; cast the recombined result to float32 afterward.
- Declared-scale luma normalization clips before widening: choose the clip dtype using `np.result_type(input_dtype, 0.0, scale)` on metadata only, clip in that dtype, then widen to float64 and divide by a device denominator. Luma restoration does not clip signed or headroom values.
- The native NLM default strength is the normalized value `10.0 / 255.0`, not either `10.0` or `255.0`. NumPy `pad(mode="reflect")` corresponds to GPU border mode `"mirror"`, not `"reflect"`.
- Native float32 scaling is qualified against OpenCV 4.11.0 with IPP AVX2 2021.12.0. Its fused vector interpolation and unfused scalar tails differ near cancellation; preserve the frozen `scale_ipp_linear` and `scale_ipp_tail_geometry` fixtures and the original signed-HDR `(9, 11) -> (7, 13)` regression. Never substitute another geometry or fixture merely because the original fails a numerical gate.
