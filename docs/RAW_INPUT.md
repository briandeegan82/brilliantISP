# RAW input contract (M2b)

The pipeline decodes Bayer input exclusively through
`util.raw_decode.RawFormat` + `decode_raw`, driven by `sensor_info` — there
is no file-size guessing, dimension factoring, tolerance matching, or
silent truncation. A payload must be exactly
`header_bytes + height * row_stride_bytes` bytes (every row, including the
last, carries full stride padding); short *and* surplus buffers are
rejected with the expected layout in the message.

## sensor_info fields

| key | meaning |
|---|---|
| `width`, `height` | full raster geometry (embedded rows included) |
| `bayer_pattern` | `rggb`/`grbg`/`gbrg`/`bggr` of the raster as stored |
| `bit_depth` | physical code depth (1..storage bits) |
| `storage_format` | `uint8`/`uint16`/`uint32`/`packed24`/`word_pair32` |
| `data_format` | legacy alias used only when `storage_format` is absent |
| `endian_type` | `ieee-le`/`ieee-be` (raw buffers only; containers keep their own byte order) |
| `data_alignment` | `LSB` (codes untouched) / `MSB` (shift `storage_bits - bit_depth`, then mask) / `MANUAL` (explicit `manual_bit_shift`) |
| `manual_bit_shift` | required for `MANUAL`, 0..storage_bits-1 |
| `header_bytes` | bytes skipped before the raster |
| `row_stride_bytes` | bytes per stored row (default `width * bytes_per_sample`) |
| `word_order` | `low_first`/`high_first` for `word_pair32` |
| `horizontal_flip` / `vertical_flip` | mirrors applied after decode/trim |
| `embedded_rows_top` / `embedded_rows_bottom` | metadata rows trimmed before flips |
| `realign_columns_from` | optional; columns `[n, width)` were read out one column early: shift right by one and refill column `n` from the same CFA colour (applied before the embedded-row trim) |
| `source_format` | `auto`/`raw`/`pgm`/`tiff`/`rawpy` |

When neither `storage_format` nor `data_format` is present, storage
defaults solely from the declared `bit_depth` (uint8 ≤ 8, uint16 ≤ 16,
uint32 otherwise) and a warning is attached to the decode result — never
from file size.

## Containers

Explicit `source_format` wins; `auto` selects by extension only
(`.raw`/`.bin`/`.dat`/none → raw, `.pgm` → pgm, `.tif`/`.tiff` → tiff,
recognized camera extensions → rawpy). Unknown or RGB image extensions
are rejected with a hint to set `source_format` explicitly — RGB images
are not Bayer input. Containers reject `header_bytes`/`row_stride_bytes`
(the container supplies its own layout) and require the declared
geometry to match the container exactly.

- **PGM**: single image, P5/P2, maxval → 8/16-bit, raster always
  big-endian regardless of `endian_type`.
- **TIFF**: one 2-D unsigned 8/16/32-bit frame only — multipage,
  multi-channel, signed, and float are rejected (channel 0 is never
  picked silently).
- **rawpy** (camera RAW): `raw_pattern` must be exactly 2x2 (XTrans
  rejected), `raw_image_visible` unsigned 8/16/32-bit, geometry equal to
  the declared size, `bit_depth` equal to `white_level.bit_length()`, LSB
  alignment only; a differing container CFA wins with a warning.

## Effective CFA

Embedded rows are trimmed before flips; the effective `bayer_pattern` is
phase-shifted by `embedded_rows_top`, then mirrored using the active
geometry (`(width-1) % 2` / `(height-1) % 2` parity). Crop applies exact
origins — including odd ones — and updates the effective CFA via the same
phase shift; `sensor_info.orig_size` records the pre-crop shape.

## Migration notes

- The old heuristic loader is retained as private
  `BrilliantISP._load_raw_legacy` for audit comparison only; it is not
  called by processing or preview paths.
- `execute(load_method=...)` other than `"auto"` now raises — declare
  `storage_format` instead.
- PGM 16-bit raster is always big-endian regardless of the configured
  `endian_type` (the container's byte order is authoritative); an explicit
  `byte_order` argument applies only to raw-buffer input.
- Crop origins — including odd ones — now apply exactly with the
  effective CFA phase updated instead of being snapped to even; combined
  with the CFA-phase correction on sensor flips, a preview or decode can
  differ from the same file run through the pre-M2b path even when the
  config is unchanged.
- `uint32` storage is decoded as standard declared byte order
  (`endian_type`); it is no longer reinterpreted as low-word-first pairs.
  Legacy big-endian `data_format: uint32` configs that relied on the old
  heuristic should set `storage_format: word_pair32` explicitly
  (`normalize_config` emits a notice).
- `estimated_raw_size` in the format stats is a RAW-layout estimate
  (`header + full raster height * row stride`), not the file length of a
  TIFF/PGM container.
- `effective_config` in `PipelineResult` and the YAML written by
  `save_pipeline_output` describe the *transformed* geometry/CFA
  (crop/flips/embedded rows) — they are not a reusable source preset;
  keep the original input config for that.
- GUI sessions preserve the original source bytes under
  `input<suffix>` (e.g. `input.pgm`); sessions staged before this change
  keep `input.raw` and are not reinterpreted.
