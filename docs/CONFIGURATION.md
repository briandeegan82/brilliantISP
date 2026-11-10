# Configuration reference (native M4 status)

This documents the configuration behavior as implemented now. The runtime is
the single native engine: every source config — `schema_version: 1` or
versionless — is migrated in memory to the native schema_version 2 mapping
at the load seam; the original source is kept on the engine as
`source_config_original`, and the migration report rides the result as
`migration_report`.

## File loading and overlays

- `util/config_merge.pipeline_config_paths(path)` keeps the legacy suffix
  rule: a bare filename is looked up under `config/`, and a `*_cam.yml` file
  is loaded **after** `config/base_hdr.yml` **if that file exists**. In the
  current tree `base_hdr.yml` lives under `config/unused/`, so the implicit
  merge does not activate; bundled `config/*.yml` files are standalone.
- `load_merged_yaml(paths)` merges documents in order with `deep_merge`:
  mappings merge recursively, lists and scalars are replaced. The result is a
  completely independent deep copy — mutating a merged config never affects
  the input mappings or another merge result.
- `load_yaml_mapping(path)` remains the raw single-file loader (no overlay or
  inheritance processing).

## Explicit `extends` inheritance (new in M1a)

Any YAML document may declare a top-level `extends` key. A runnable example
ships in the tree as `config/experiment.yml`:

```yaml
extends: SVS_cam.yml
platform:
  debug_enabled: false
```

It inherits the full `config/SVS_cam.yml` and overrides one platform flag.
For inheriting several bases, the list form is merged in listed order before
the child:

```yaml
extends: [first_base.yml, second_base.yml]
```

Both entries must be real files relative to the referring document; mixing
two independent camera presets is not recommended (later sensor geometry,
CFA, and calibration values silently win).

Rules:

- Entries are path strings resolved **relative to the referring file**
  (`~` expansion allowed). Absolute paths are permitted.
- Inherited documents are merged in listed order before the child; the
  `extends` key is removed from the effective config.
- Inheritance is recursive: a base may itself `extends` another file.
- The same base may appear in multiple branches (diamond inheritance is
  legal); a true cycle raises `ValueError` naming the full path chain.
- `extends` must be a non-empty path string or non-empty list of path
  strings; anything else raises `ValueError`.
- Only explicit paths are honored — there is no filename guessing for
  `extends`.

## Shared metadata (`util/config_schema.py`)

- With both BLC and PWC enabled, BLC may subtract black in the sensor-code
  domain only if PWC's pre/post black levels are zero and BLC `is_linear` is
  false. The validator rejects overlapping subtraction or BLC rescaling
  before the PWC input-code LUT. `pre_linearization_black_level` takes
  precedence over the legacy `pedestal` alias. This is a safety check, not
  a change to stage order or a calibration of BLC offsets for PWC output.
- `REQUIRED_CONFIG_KEYS` is the single source for the required
  top-level sections of a *v1* source; `BrilliantISP._REQUIRED_CONFIG_KEYS`
  and `tools.gui.yaml_compat.REQUIRED_CONFIG_KEYS` alias it.
  `NATIVE_REQUIRED_CONFIG_KEYS` is the native v2 set: it drops the four
  retired color sections (`gamma_correction`, `color_space_conversion`,
  `rgb_conversion`, `yuv_conversion_format`) and adds `output`. A v2 source
  that still *enables* a retired section is rejected; merely present
  disabled/legacy keys are recorded in the migration report only.
  Output constraints (`output.model/color_space/transfer/gamma/bit_depth/
  format/matrix/range/layout/packing`) are owned by
  `OutputSpec.from_mapping`, not by ad-hoc validation. v2 white-balance
  gains are nonnegative — explicit zero is allowed and simply attenuates
  the channel; suggestions-only AWB is `white_balance.is_auto: false`
  instead. v1 keeps the positive-gain rule. All 13 demosaic algorithm
  enums are shared between schema and UI.
- Frozen `ParameterSpec` / `BlockSpec` descriptors seed a partial schema.
  Defaults are copied literally from `config/unused/base_hdr.yml` or marked
  `ABSENT` when the base file does not define them (e.g. the commented-out
  `exposure_correction_mode` keys). Coverage currently includes the fields
  the GUI validators consume plus the explicit enums the GUI exposes; it is
  not yet a complete per-block schema.
- `BLOCK_SPECS` groups parameters per required top-level section; `doc_page`
  entries point at `docs/blocks/` pages — `tone_mapping.md` exists, the
  rest are still planned.

## Standalone validator CLI

`tools/validate_config.py` validates configs without importing the pipeline,
Tk, or GPU modules — usable from the repo root or any working directory:

```bash
python tools/validate_config.py --config config/SVS_cam.yml
python tools/validate_config.py --config config/experiment.yml --json
python tools/validate_config.py --config config/SVS_cam.yml --config my_overlay.yml
```

`--config` is repeatable and merges in argument order (later files override
earlier ones) after `pipeline_config_paths` resolution and `extends`
inheritance. Output is either human-readable text plus `Configuration valid`
on success, or a single JSON object `{valid, config_sources, notices,
issues:[{path,message,severity}]}`. Exit codes: `0` valid (warnings allowed),
`1` validation errors, `2` load/parse/normalization errors (reported as an
actionable `error:` message or `{"valid": false, "error": ...}` JSON, never a
traceback for expected input problems). `schema_version: 2` sources are
supported — a sparse v2 config fails as exit `1` (missing `output` etc.), not
as an unsupported-version load error. The tool checks the version-appropriate
required-key set plus `validate_config` rules; it does not instantiate the
pipeline.

## `normalize_config(mapping)`

Returns `(normalized_mapping, notices)` with a deep copy — the input is never
mutated and no defaults are inserted.

- Versionless mappings, `schema_version: 1`, and `schema_version: 2` are
  accepted (one notice is emitted for versionless configs). Note that
  `normalize_config` itself does **not** migrate — the in-memory v1→v2
  migration happens at the `BrilliantISP.load_config` seam after
  normalization. An
  explicitly null `schema_version` is an invalid integer value and raises
  `ValueError`, same as any non-integer or `>2`.
- Unknown top-level keys are **retained** and reported as warning notices.
- Precise warnings are emitted for configuration that the shared
  operators do not consume: `aces.variant`, `aces.output_transform`, `aces.surround`,
  `aces.white_point`, and `*.dark_enh` in the integer TMO sections
  (`reinhard_integer`, `aces_integer`, `hable_integer`, `integer_tmo`).
  `aces.variant: full`/`canonical` is flagged as selecting an ACES variant
  that does not exist — only the fitted curve is implemented.

## `validate_config(config, require_raw=False, has_raw=False)`

Shared numeric/geometry validation used by the calibration GUI
(`tools/gui/validation.py` is a compatibility wrapper). Issues are
`ValidationIssue(path, message, severity)` objects; `severity="error"` blocks
processing via `can_process`, `"warning"` is advisory only.

Errors: positive-integer `sensor_info.width`/`height`; `bayer_pattern` in
`rggb/grbg/gbrg/bggr`; integral `bit_depth` 1–32; `endian_type` in
`ieee-le`/`ieee-be`. `sensor_info` additionally accepts explicit RAW-layout
fields — `storage_format` (overrides legacy `data_format`),
`source_format`, `header_bytes`, `row_stride_bytes`, `word_order`,
`embedded_rows_top`/`embedded_rows_bottom`, `manual_bit_shift`, and
`data_alignment` incl. `MANUAL`. The full input contract and migration
notes are documented in `docs/RAW_INPUT.md`; crop origin bounds are
validated against the active (post-embedded) height, and odd origins are
now applied exactly with the CFA phase updated (warning only notes the
changed phase semantics vs. the legacy snap). Also checked: finite
`white_balance.r_gain`/`b_gain` (positive in v1; nonnegative in v2);
`digital_gain.gain_array` a list of positive finite strictly increasing
gains; `digital_gain.current_gain` an in-range integer index whose selected
gain is positive; enabled `crop` must have positive integral size inside the
sensor frame, and explicitly supplied crop origins must be integral (absent
origins default to 0 as before). Booleans are rejected as numbers; NaN/inf
and fractional geometry/indices are rejected; malformed sections report
issues instead of raising.

Warnings (non-blocking): an **odd crop origin** is applied exactly with
the CFA phase updated (the warning only notes the changed phase semantics
vs. the legacy snap-down behavior).

Note: `config/triton_lauren_cam.yml` and
`config/triton_lauren_hdr24_cam.yml` now spell little-endian as the canonical
`ieee-le` (previously `little`, which the pipeline already decoded
identically). Other legacy alias spellings in external files remain a
planned migration concern; the validator enum was not widened.

## Entry-point guard

`normalize_config` runs at both real loader seams: `BrilliantISP.load_config`
stores the notices on `self.config_notices` (for future UI/CLI display) and
`tools/gui/yaml_compat.load_preset_yaml` returns the normalized mapping.
Invalid `schema_version` (explicit `null`, non-integer, `>2`) raises
`ValueError` everywhere; a valid v2 source loads natively in the pipeline,
but the GUI preset loader rejects v2 editing (directs to the v1 source) and
the boltISP exporter rejects v2 as well — v1 GUI/export workflows are
unchanged.

## Migration paths

Tone-mapping migration is active in `tools/migrate_tone_mapping.py`:
it emits canonical schema-version-1 settings and a separate preservation
report without overwriting its sources or existing destinations.
Canonical aliases (`aces`/`aces_integer` → `aces_fitted` + backend, etc.)
are supported now; see the tone-mapping reference below.

`tools/migrate_pipeline_config.py` emits the migrated native
`schema_version: 2` config plus a migration report: repeatable `--config`
overlays (resolved via `pipeline_config_paths`/`extends`), required
`--output`/`--report` exclusive destinations (parents must exist, paths
distinct, existing files/symlinks rejected, payloads serialized before
writing, rollback removes only files it created), sources never written.
`tools/migrate_tone_mapping.py` remains the v1 tone-only migration CLI.
The historical `config/unused/base_hdr.yml`/`base_sdr.yml` stay unused —
they are not restored as active defaults.

## Tone mapping

`tone_mapping` accepts either the legacy selector form
(`tone_mapper: durand|hdr_durand|aces|aces_integer|hable|hable_integer|reinhard_integer|integer_tmo` plus
`tone_mapping_before_demosaic`) or a canonical form
(`operator`, required `backend`, optional `placement`, `normalization`,
`exposure_ev`, `domain_scale`, `normalize_output`, `parameters`,
`lut_domain`, `lut_size`). Canonical values take precedence with a
notice; `backend` is mandatory when `operator` is present and processing
is enabled. `hdr_durand` and the other legacy operator sections remain
optional known sections; the old `aces.output_transform` enum is removed
(legacy fields are ignored with notices). Migration:
`tools/migrate_tone_mapping.py` produces a canonical schema-v1 config
plus report. Full contract: `docs/blocks/tone_mapping.md`. The native
post-CCM RGB path also reads `rgb_norm`, `rgb_norm_k`, `near_white_desat`
and `adaptive_white`; see `docs/ISP_BLOCKS_AND_TUNING.md`.
