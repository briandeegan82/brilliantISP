# Tone mapping (M3 unified engine)

The tone-mapping stage is a shared engine in `modules/tone_mapping/`;
every legacy wrapper delegates to it.

## Declared semantics

- `ToneMappingSpec` (registry.py) is a frozen, validated, hashable
  description: `operator`, `backend`, `normalization`, `exposure_ev`,
  `domain_scale`, `normalize_output`, `parameters`, `lut_domain`,
  `lut_size`, `enabled`.
- Operators/backends: `aces_fitted` (float/lut), `hable` (float/lut),
  `reinhard` (float/lut/fixed), `durand` (cpu only, local operator).
- Math order: normalize once -> gain `2**exposure_ev * domain_scale` ->
  saturate curve input at `lut_domain` -> scalar curve -> optional
  theoretical output normalization -> clip [0,1] -> output code scale.
- `input_max` is the explicit declared code maximum (never inferred from
  pixel data), positive and at most `2**32-1`; `output_max` is an integer
  code maximum in `1..65535`.

## Scalar definitions and parameter defaults

All global curves clip their result to `[0,1]`. Let `x` be the nonnegative
curve input after normalization, exposure/domain gain, and domain saturation.

| Operator | Definition | Parameters |
|---|---|---|
| `aces_fitted` | `x(2.51x + .03) / (x(2.43x + .59) + .14)` | None |
| `hable` | `H(exposure_bias*x) / H(white_point)` | `exposure_bias=2`, `white_point=11.2` |
| `reinhard` | `x / (x + knee/strength)` | `knee=.25`, `strength=1` |

The standard Hable partial curve is
`H(z) = (.15z² + .05z + .004)/(.15z² + .5z + .06) - 1/15`.
The bias is a **linear multiplier**, not EV; `exposure_ev` is a separate
power-of-two multiplier. Parameters must be positive, finite, and
numerically representable by the selected implementation.

`aces_fitted` is a scalar fitted approximation, not a full ACES color
pipeline or certified RRT/ODT. There is no internal gamma/OETF in any
operator — the single output transfer (`srgb`/`rec709`/`power`/`linear`)
is owned by the output stage, implemented in the native engine.

## Backends

- `float`: float64 end-to-end; allows finite headroom above `input_max`.
- `lut`: table of the same curve (default 65536 entries, power of two
  256..65536), with the same normalization, clipped and rounded to
  output codes. The table covers `[0,lut_domain]` normally. With
  `normalize_output=true`, it covers `[0,min(gain,lut_domain)]`: values
  above that reference already saturate to one mathematically. This
  preserves table resolution at small exposures rather than collapsing
  normalized output to black. Integer sources use uint64 Q32 coordinates
  and Q16 linear interpolation; float sources interpolate in float64.
  Tables are cached in a shared 8 MiB
  immutable-byte LRU keyed by the complete mathematical config.
- `fixed`: exact integer Q32 Reinhard; unsigned uint8/16/32 input only.
- `cpu` (durand): Gaussian/range local operator (SciPy gaussian_filter);
  frame-adaptive output normalization; `normalize_output` rejected.

Both float and LUT execution saturate curve input at `lut_domain`
(default 16). This is a finite-domain approximation, not unlimited
floating-point headroom. EV is bounded to `[-32,32]`; scalar gains,
normalization, and references must remain representable. LUT error depends
on size, domain, and curve parameters; a tested error bound is not a
universal guarantee for arbitrary steep curves or smaller tables.

Fixed Reinhard uses `M=2**32-1`, saturating normalized Q32 coordinates, and
the shared `knee/strength` ratio. That ratio is evaluated before dividing by
effective gain, avoiding spurious underflow/overflow of `strength*gain`.
Image products are widened before multiplication.
It accepts unsigned integer sources only. Post-CCM RGB uses the exact
numerator `2126R+7152G+722B` with input range multiplied by 10000, preserving
fractional luminance instead of rounding it to a code first. Direct
`map_luminance` callers with float RGB must select float or LUT execution;
the native pipeline instead reaches `fixed` through the Q8.24 bridge
described below.

## Normalization

- `declared_range`: `x / input_reference` (defaults to `input_max`;
  headroom retained until the domain cap).
- `per_frame`: whole-frame or supplied `FrameStats` extrema; constant
  frames fall back to declared range instead of all-black; externally
  supplied stats are reused unchanged for ROI/tile calls.

The global API is:

```python
mapped = map_luminance(
    luminance, input_max=16777215, output_max=65535,
    config=spec, frame_stats=None,
)
```

`map_luminance` lives in `modules.tone_mapping.backends`; `ToneMappingSpec`
in `registry`. Float execution and float-input LUT execution return
float64 **output-code units**, not normalized values. Unsigned-input LUT
and fixed execution return uint16 code values. Unsigned sources are
uint8/16/32 and must not exceed their declared input maximum. Float sources
may have finite headroom; declared-range normalization bounds negative
values at zero. Empty, complex, boolean, and nonfinite active inputs fail.

The native engine calls `modules.tone_mapping.native.map_native` on
normalized (0..1, headroom-preserving, signed) F32 data — a nonempty 2-D
Bayer plane or HWC3 RGB. For RGB it maps the Rec.709 luminance through
the same operator/backend machinery and rescales RGB by the
`mapped_y / positive_y` ratio (zero or nonpositive-Y pixels map to black),
preserving the per-pixel RGB/luminance ratios without a gamut clip.
Float input is supported for the `float`/`lut` backends directly and for
`fixed` through the explicit Q8.24 integer bridge — the selected
normalization runs first, then saturation at `u_cap` with named errors on
overflow — so the integer-only restriction applies to the direct
`map_luminance` API, not to the native pipeline entry.

`FrameStats` preserves exact integer or float extrema in input-code units.
An optional `input_reference` declares the normalization white separately
from the `input_max` storage cap (defaults to `input_max`); declared
normalization divides by the reference when provided.
Global ROI calls can reuse whole-frame extrema. This does not make local
Durand tiled/ROI execution equivalent: it also depends on spatial context
and whole-output extrema. No local tiled-parity claim is made.

Disabled execution bypasses operator math, inactive parameter validation,
LUT construction, and diagnostics. It returns the same object if ranges
match; otherwise it performs the required clipped range conversion. The
pipeline adapter returns unchanged post-CCM data when disabled and scales
declared HDR to declared pipeline codes before demosaic.

## Local Gaussian/range approximation

Defaults are `sigma_space=2`, `sigma_color=.4`, `contrast_factor=2`,
`downsample_factor=1`. On normalized/gained/domain-bounded input,
the float32 log plane is `logY=log10(x+1e-6)`. With Gaussian output `g`,
`d=logY-g`, the base is `g+exp(-.5(d/sigma_color)²)*d`; reconstruction is
`10**(base/contrast_factor + logY-base)`. SciPy Gaussian filtering uses
reflect boundaries and its default truncate 4. This is not an exact bilateral
filter.

Explicit canonical downsampling resizes the log plane with area filtering
to `(ceil(W/f),ceil(H/f))`, at least one pixel per dimension; spatial sigma
is divided by `f`, the base is resized back with bilinear filtering, and
the original shape is restored exactly. Legacy factors were unused and
resolve to 1 with a notice; only canonical configuration enables this
approximation. An injected Gaussian callback is a testing/future accelerator
seam, not verified GPU support.

## Intentional changes vs legacy

- ACES is `aces_fitted` only — the internal OETF/gamma stage was removed;
  exposure applies AFTER normalization. Legacy `aces` resolves to
  `domain_scale=100` (lut_domain stays the shared 16 default).
- Legacy `hable`/`hable_integer` never consume EV (`exposure_ev=0`);
  `exposure_bias` wins, with `exposure_adjust` as the LINEAR alias only.
- No bit-depth inference from observed maxima anywhere.
- Integer paths clip before the unsigned cast; float paths stay float64.
- RGB luminance uses a widened weighted sum before division. Declared RGB
  code ranges replace HDR-range assumptions after CCM. Some old integer/RGB
  and dark-HDR outputs therefore change substantially, not merely by an
  interpolation LSB; keeping a legacy selector does not retain those bugs.

## Constant frames and diagnostic storage

- Constant per-frame normalized input falls back to declared range BEFORE
  the curve for global operators (saturated output can still legitimately
  be bright); the local Durand's constant *output* falls back to the
  bounded normalized input.
- Native `.npy` mapped saves are an explicit M3 intermediate-format
  policy (`diagnostics.save_mapped_output`): native NPY preserves
  precision independently of the final save format and always targets
  `platform.output_dir` — the old `save_output_array` path is not used
  by tone mapping.

## Config resolution

Save this example at the repository root as `my_tmo.yml`; it inherits the
existing camera preset without overwriting it:

```yaml
schema_version: 1
extends: config/SVS_cam.yml
tone_mapping:
  is_enable: true
  operator: hable        # aces_fitted|hable|reinhard|durand
  backend: lut           # float|lut|fixed|cpu (required when operator set)
  placement: post_ccm    # pre_demosaic|post_ccm
  normalization: declared_range   # declared_range|per_frame
  exposure_ev: 0.0
  domain_scale: 1.0
  normalize_output: false
  parameters: {exposure_bias: 2.0, white_point: 11.2}
  lut_domain: 16.0
  lut_size: 65536        # power of two 256..65536
```

| Legacy selector | Selected section | Operator/backend | Normalization/domain |
|---|---|---|---|
| `aces` | `aces` | `aces_fitted/float` | per-frame; domain multiplier 100 |
| `aces_integer` | `aces_integer` | `aces_fitted/lut` | per-frame by default; `hdr_scale` default 1 |
| `hable` | `hable` | `hable/float` | declared range; multiplier 1 |
| `hable_integer` | `hable_integer` | `hable/lut` | per-frame by default; `hdr_scale` default 1 |
| `reinhard_integer` | `reinhard_integer`, fallback `integer_tmo` | `reinhard/fixed` | declared range; multiplier 1 |
| `integer_tmo` | `integer_tmo`, fallback `reinhard_integer` | `reinhard/fixed` | declared range; multiplier 1 |
| `durand` or `hdr_durand` | `hdr_durand` | `durand/cpu` | declared range; multiplier 1 |

Integer global variants preserve `normalize_output` (default false).
`use_normalization=false` selects declared range for ACES/Hable LUT
variants. Both Hable variants use `exposure_bias`, falling back to
`exposure_adjust`, default 2. Float ACES EV uses `exposure_adjust`, falling
back to `exposure_adjustment`; integer ACES reverses that precedence.
Inactive fallback aliases are not evaluated.

`resolve_tone_mapping(full_config)` returns
`ResolvedToneMapping(spec, placement, is_save, is_plot_curve, notices,
legacy_section)`. Canonical `tone_mapping.operator` (requires explicit
`backend`) wins over the legacy `tone_mapper` selector; canonical
`placement` wins over `tone_mapping_before_demosaic` (default
`post_ccm`). Legacy selectors `durand`, `hdr_durand`, `aces`,
`aces_integer`, `hable`, `hable_integer`, `reinhard_integer`,
`integer_tmo` resolve as listed above. Selected legacy `is_enable` combines
with the global flag. Canonical unknown fields/parameters fail rather than
silently appearing functional. Selected full/canonical ACES fails; legacy
`gamma`, output-transform, surround, white-point, and `dark_enh` fields
that no longer affect the selected implementation are reported as ignored.
Inactive legacy choices do not invalidate another active operator.

## Migration and export

- `tools/migrate_tone_mapping.py --config a.yml --output out.yml --report report.json`
  produces a schema-version-1 canonical config and
  a JSON report; both targets must be new files (no overwrites, sources
  untouched). Legacy sections are preserved in the report
  (`legacy_sections` = all originals, `inactive_presets` = unselected);
  `source_schema_version` reports the actual field (None when absent) and
  the output always sets `schema_version: 1`. Conversion validates selected
  settings even if currently disabled, without enabling them in the output.
  Unconvertible disabled selections must be corrected before migration;
  runtime disabled processing still ignores their inactive controls.
  Canonical migration materializes defaults and removes remaining legacy
  sections; subsequent conversion is idempotent. The report also retains the
  original `tone_mapping` selection. Repeatable `--config` and recursive
  `extends` use the normal loader. Both destinations are exclusive-created;
  invalid configurations produce neither output.
- `tools/export_boltisp_yaml.py` translates only the canonical
  **Hable** subset (verified `hable_*` keys); canonical Durand,
  ACES and Reinhard are rejected clearly rather than emitting
  misleading keys. All legacy `tone_mapper` exports remain
  unchanged. Approximation warnings are emitted; no external
  parity is claimed.

## Remaining gaps

- Full ACES RRT+ODT, canonical/full variants: not implemented.
- Final output transfer/OETF is now implemented at the output contract
  (`encode_transfer` + `OutputSpec`); it was the last unowned stage.
- GPU Durand: unverified, never auto-selected (CPU fallback only).
- Legacy wrapper classes: removal target M8.

The pipeline adapter retains
`execute(visualize_output=True, save_output=True, *, plot_curve=None)`.
`save_output=false` prevents **all** artifacts, including plots. Native NPY
saves additionally require resolved `is_save`; plots require visualization
and their resolved flag or explicit plot request. Existing NPY/PNG paths are
not overwritten: repeated saves allocate an exclusive numbered suffix,
including repeated 3A passes on the same source; when metadata is supplied
the NPY+JSON pair is reserved as one collision-safe group. Failed writes
remove only the newly created partial files. Curves use local Agg figures
without changing Matplotlib's global backend. Compute-only
`ProcessingOptions` suppress these flags.

In the native engine the diagnostic emits exactly one normalized F32 NPY +
`diagnostic_schema_version: 1` JSON for the actual final TMO stage
(`tone_mapping_pre` for the advanced branch, `tone_mapping_post`
otherwise) after the AE controller finishes — never per meter pass — with
signal-domain `tone_mapped_bayer`/`display_linear_rgb`, linear transfer,
and the resolved spec serialized verbatim.

Useful commands from the repository root:

```bash
venv/bin/python tools/validate_config.py --config my_tmo.yml --json
venv/bin/python tools/migrate_tone_mapping.py \
  --config config/SVS_cam.yml \
  --output tmp/SVS_m3.yml --report tmp/SVS_m3_report.json
```

The migration destinations must not already exist. No shipped camera preset
is rewritten automatically. Schema-v2 migration is now implemented
(`tools/migrate_pipeline_config.py`); measured CPU optimization and desktop
GUI validation remain separate milestones.

## M3 validation checkpoint (historical)

Historical M3 evidence (not current quality proof):
the full software suite passed 468 tests, followed by a final 144-test
tone-mapping gate including live-alias, repeated-3A-save, partial-write
cleanup, unsigned-RGB-range, and combined fixed-parameter regressions.
Type checking passed for maintained M3 files and tests.
An independent 3606-case numerical grid covered integer
8/12/16/24/32-bit domains, normalization choices, exposure values including
-32 and -20 EV, and backend variants. Maximum errors on that grid were below
one 16-bit output code for integer LUT, about 0.506 code for float-input LUT,
and about 0.501 code for fixed execution. These are finite-grid measurements,
not universal approximation guarantees.

Both private frozen scenes have exact facade/API agreement and unchanged
upstream inputs. At the active stage the SVS pre-demosaic result was within
one code of an independent float64 reference; the AD post-CCM result matched
its independently quantized RGB reference exactly. The AD preview remained
bit-exact against the old pipeline. The SVS preview changed: RMSE 0.510 and
maximum 28 codes on its 8-bit output. Its 16-bit tone-mapped values differ from
the legacy implementation by at most 5 codes; the M3 early 8-bit/YUV/OETF
path amplified small native changes. This historical result is not a claim
of perceptual image-quality improvement; the current native path owns
precision/transfer explicitly and still requires its own quality gate.

Local evidence is retained under `tmp/modernization_m0/` as
`m3_final_tests.txt`, `m3_final_types.txt`, `m3_scalar_final_tests.txt`,
`m3_scalar_final_types.txt`, `m3_numerics_accepted.json`, and `m3_after.json`.
Earlier numerical reports remain frozen separately.
Frozen inputs, old baselines, and the four pre-existing
modified diagnostic PNGs were hash-checked unchanged. No CPU speedup or
external boltISP parity is claimed by this milestone.
