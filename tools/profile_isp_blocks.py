#!/usr/bin/env python3
"""Profile BrilliantISP block execution times and write a report."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from brilliant_isp import BrilliantISP  # noqa: E402
from util.config_merge import load_merged_yaml, pipeline_config_paths  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "svs_cam.yml"
DEFAULT_RAW = REPO_ROOT / "in_frames" / "hdr_mode" / "frame_0460_fsin_38361194647660880.raw"
DEFAULT_REPORT = REPO_ROOT / "reports" / "isp_block_profile_report.md"
DEFAULT_DATA = REPO_ROOT / "reports" / "isp_block_profile_data.json"

PIPELINE_BLOCKS: list[dict[str, str]] = [
    {"key": "crop", "name": "Crop", "logger": "Crop"},
    {
        "key": "dead_pixel_correction",
        "name": "Dead Pixel Correction",
        "logger": "DeadPixelCorrection",
    },
    {
        "key": "black_level_correction",
        "name": "Black Level Correction",
        "logger": "BlackLevelCorrection",
    },
    {"key": "companding", "name": "Companding / PWC", "logger": "PWC"},
    {"key": "oecf", "name": "OECF", "logger": "OECF"},
    {"key": "digital_gain", "name": "Digital Gain", "logger": "DigitalGain"},
    {
        "key": "lens_shading_correction",
        "name": "Lens Shading Correction",
        "logger": "LensShadingCorrection",
    },
    {
        "key": "bayer_noise_reduction",
        "name": "Bayer Noise Reduction",
        "logger": "BayerNoiseReduction",
    },
    {
        "key": "auto_white_balance",
        "name": "Auto White Balance",
        "logger": "AutoWhiteBalance",
    },
    {"key": "white_balance", "name": "White Balance", "logger": "WhiteBalanceOptimized"},
    {"key": "tone_mapping", "name": "Tone Mapping", "logger": "ToneMapping"},
    {"key": "demosaic", "name": "Demosaic", "logger": "Demosaic"},
    {
        "key": "color_correction_matrix",
        "name": "Color Correction Matrix",
        "logger": "ColorCorrectionMatrixOptimized",
    },
    {"key": "auto_exposure", "name": "Auto Exposure", "logger": "AutoExposure"},
    {
        "key": "linear_16bit_to_8bit",
        "name": "Linear 16-bit to 8-bit Conversion",
        "logger": "InlineStep",
    },
    {
        "key": "color_space_conversion",
        "name": "Color Space Conversion",
        "logger": "ColorSpaceConversion",
    },
    {"key": "ldci", "name": "LDCI", "logger": "LDCI"},
    {"key": "sharpen", "name": "Sharpen", "logger": "Sharpening"},
    {"key": "2d_noise_reduction", "name": "2D Noise Reduction", "logger": "NoiseReduction2d"},
    {"key": "rgb_conversion", "name": "RGB Conversion", "logger": "RGBConversion"},
    {"key": "gamma_correction", "name": "Gamma Correction", "logger": "GammaCorrection"},
    {"key": "scale", "name": "Scale", "logger": "Scale"},
    {"key": "yuv_conversion_format", "name": "YUV Conversion Format", "logger": "YUVConvFormat"},
]

LOGGER_TO_KEY = {entry["logger"]: entry["key"] for entry in PIPELINE_BLOCKS}
KEY_TO_NAME = {entry["key"]: entry["name"] for entry in PIPELINE_BLOCKS}
KNOWN_TONE_MAPPERS = {
    "HDRDurandToneMapping": "tone_mapping",
    "ACESToneMapping": "tone_mapping",
    "IntegerReinhardToneMapping": "tone_mapping",
    "ACESIntegerToneMapping": "tone_mapping",
    "HableToneMapping": "tone_mapping",
    "HableIntegerToneMapping": "tone_mapping",
}

EXECUTION_TIME_RE = re.compile(
    r" - (?P<logger>[A-Za-z0-9_]+) - INFO -\s+(?:(?:LSC )?[Ee]xecution time): (?P<seconds>[0-9.]+)s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per variant.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs per variant before recording stats.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--single-run-config", type=Path, default=None)
    parser.add_argument("--single-run-raw", type=Path, default=None)
    parser.add_argument("--single-run-variant", default="single_run")
    return parser.parse_args()


def deep_disable_outputs(config: dict[str, Any]) -> None:
    """Disable expensive side outputs that are not part of timing."""
    for key, value in list(config.items()):
        if isinstance(value, dict):
            deep_disable_outputs(value)
            if "is_save" in value:
                value["is_save"] = False
            if "is_plot_curve" in value:
                value["is_plot_curve"] = False


def build_variant_config(
    base_config: dict[str, Any],
    *,
    log_path: Path,
    enabled_overrides: set[str] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    enabled_overrides = enabled_overrides or set()

    config["platform"]["debug_enabled"] = True
    config["platform"]["debug_log_level"] = "INFO"
    config["platform"]["debug_log_file"] = str(log_path)
    config["platform"]["plot_histograms"] = False
    config["platform"]["histogram_show_log"] = False
    config["platform"]["histogram_show_channels"] = False
    config["platform"]["disable_progress_bar"] = True
    config["platform"]["render_3a"] = False
    config["platform"]["skip_disabled_modules"] = False

    deep_disable_outputs(config)

    for key in enabled_overrides:
        if key in config and isinstance(config[key], dict) and "is_enable" in config[key]:
            config[key]["is_enable"] = True

    return config


def write_temp_config(config: dict[str, Any], temp_dir: Path, suffix: str) -> Path:
    config_path = temp_dir / f"profile_{suffix}.yml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def clear_logger_handlers() -> None:
    """Avoid handler duplication across repeated in-process runs."""
    all_logger_names = set(LOGGER_TO_KEY) | set(KNOWN_TONE_MAPPERS) | {"BrilliantISP"}
    for logger_name in all_logger_names:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def parse_timings(log_text: str) -> dict[str, float]:
    timings: dict[str, float] = {}

    for line in log_text.splitlines():
        match = EXECUTION_TIME_RE.search(line)
        if match:
            logger_name = match.group("logger")
            seconds = float(match.group("seconds"))
            key = LOGGER_TO_KEY.get(logger_name) or KNOWN_TONE_MAPPERS.get(logger_name)
            if key:
                timings[key] = seconds

    return timings


def run_single_pipeline(config_path: Path, raw_path: Path, variant_name: str) -> dict[str, Any]:
    clear_logger_handlers()
    data_path = str(raw_path.parent)
    isp = BrilliantISP(data_path, str(config_path), outFileName=f"profile_{variant_name}")
    isp.raw_file = raw_path.name
    if isp.c_yaml is None:
        raise RuntimeError("Config failed to load.")
    isp.c_yaml["platform"]["filename"] = raw_path.name

    byte_order = isp.sensor_info["endian_type"] if isp.sensor_info else "ieee-le"
    load_byte_order = "big" if "be" in byte_order else "little"
    isp.load_raw(byte_order=load_byte_order)

    start = time.perf_counter()
    isp.run_pipeline(visualize_output=False)
    total_seconds = time.perf_counter() - start

    return {
        "wall_time_seconds": total_seconds,
    }


def run_pipeline_once(config_path: Path, raw_path: Path, variant_name: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run-config",
        str(config_path),
        "--single-run-raw",
        str(raw_path),
        "--single-run-variant",
        variant_name,
    ]
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    combined_output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
    marker_match = re.search(r"__PROFILE_RESULT__(\{.*\})", combined_output)
    if marker_match is None:
        raise RuntimeError("Single-run profiler did not emit a result marker.\n" f"Captured output:\n{combined_output}")

    result = json.loads(marker_match.group(1))
    result["block_timings_seconds"] = parse_timings(combined_output)
    return result


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    block_samples: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for key, value in run["block_timings_seconds"].items():
            block_samples[key].append(value)

    summary = {
        "wall_time_seconds": {
            "mean": mean([run["wall_time_seconds"] for run in runs]),
            "stdev": stdev([run["wall_time_seconds"] for run in runs]),
            "min": min(run["wall_time_seconds"] for run in runs),
            "max": max(run["wall_time_seconds"] for run in runs),
        },
        "blocks": {},
    }
    for key, samples in block_samples.items():
        summary["blocks"][key] = {
            "mean": mean(samples),
            "stdev": stdev(samples),
            "min": min(samples),
            "max": max(samples),
            "runs_observed": len(samples),
        }
    return summary


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    *,
    config_path: Path,
    raw_path: Path,
    runs_per_variant: int,
    warmup_runs: int,
    base_config: dict[str, Any],
    variant_results: dict[str, dict[str, Any]],
) -> str:
    base_summary = variant_results["base"]["summary"]
    base_enabled = []
    base_disabled = []
    for block in PIPELINE_BLOCKS:
        key = block["key"]
        cfg = base_config.get(key)
        if key == "linear_16bit_to_8bit":
            base_enabled.append(block["name"])
        elif isinstance(cfg, dict) and "is_enable" in cfg and cfg["is_enable"]:
            base_enabled.append(block["name"])
        elif isinstance(cfg, dict) and "is_enable" in cfg:
            base_disabled.append(block["name"])
        else:
            base_enabled.append(block["name"])

    base_rows: list[list[str]] = []
    for block in PIPELINE_BLOCKS:
        key = block["key"]
        stats = base_summary["blocks"].get(key)
        share_text = "n/a"
        if stats and base_summary["wall_time_seconds"]["mean"] and stats["mean"] is not None:
            share = 100.0 * stats["mean"] / base_summary["wall_time_seconds"]["mean"]
            share_text = f"{share:.1f}%"

        if key == "linear_16bit_to_8bit":
            status = "Inline step"
        else:
            cfg = base_config.get(key)
            if not isinstance(cfg, dict) or "is_enable" not in cfg:
                status = "Enabled"
            elif cfg["is_enable"] and stats is None:
                status = "Enabled, not timed"
            elif cfg["is_enable"]:
                status = "Enabled"
            else:
                status = "Disabled"

        base_rows.append(
            [
                KEY_TO_NAME.get(key, key),
                status,
                format_seconds(stats["mean"] if stats else None),
                format_seconds(stats["stdev"] if stats else None),
                format_seconds(stats["min"] if stats else None),
                format_seconds(stats["max"] if stats else None),
                share_text,
            ]
        )

    isolated_rows: list[list[str]] = []
    for variant_name, result in variant_results.items():
        if variant_name == "base":
            continue
        key = result["focus_block"]
        stats = result["summary"]["blocks"].get(key, {})
        wall_stats = result["summary"]["wall_time_seconds"]
        delta = None
        if wall_stats["mean"] is not None and base_summary["wall_time_seconds"]["mean"] is not None:
            delta = wall_stats["mean"] - base_summary["wall_time_seconds"]["mean"]

        isolated_rows.append(
            [
                KEY_TO_NAME[key],
                format_seconds(stats.get("mean")),
                format_seconds(stats.get("stdev")),
                format_seconds(wall_stats["mean"]),
                f"{delta:+.3f}s" if delta is not None else "n/a",
            ]
        )

    methodology = [
        f"- Config: `{config_path}`",
        f"- Input: `{raw_path}`",
        f"- Measured runs per variant: `{runs_per_variant}`",
        f"- Warm-up runs per variant: `{warmup_runs}`",
        "- Benchmark path: `BrilliantISP.run_pipeline(visualize_output=False)`",
        "- Side outputs disabled during profiling: histogram plots, tone-curve plots, and module save outputs",
        "- The inline 16-bit to 8-bit conversion step is included in coverage but left unmeasured because the current code does not emit a dedicated high-resolution timing event for it",
    ]

    observed_blocks = [(key, stats) for key, stats in base_summary["blocks"].items() if stats["mean"] is not None]
    slowest_block_summary = max(observed_blocks, key=lambda item: item[1]["mean"]) if observed_blocks else None

    summary_lines = [
        f"- Mean end-to-end wall time: `{format_seconds(base_summary['wall_time_seconds']['mean'])}`",
        f"- Run-to-run stdev: `{format_seconds(base_summary['wall_time_seconds']['stdev'])}`",
        (
            (
                f"- Slowest measured block in base config: "
                f"`{KEY_TO_NAME[slowest_block_summary[0]]}` at "
                f"`{format_seconds(slowest_block_summary[1]['mean'])}`"
            )
            if slowest_block_summary
            else "- No block timings captured"
        ),
        f"- Enabled in base config: `{', '.join(base_enabled)}`",
        (
            f"- Disabled in base config but profiled separately: `{', '.join(base_disabled)}`"
            if base_disabled
            else "- No disabled blocks required separate profiling"
        ),
    ]

    return "\n".join(
        [
            "# ISP Block Execution Time Profile",
            "",
            "## Summary",
            *summary_lines,
            "",
            "## Methodology",
            *methodology,
            "",
            "## Base Config Profile",
            markdown_table(
                ["Block", "Status", "Mean", "Stdev", "Min", "Max", "Share of Pipeline"],
                base_rows,
            ),
            "",
            "## Disabled Block Spot Checks",
            markdown_table(
                [
                    "Block",
                    "Block Mean",
                    "Block Stdev",
                    "Variant Wall Time",
                    "Delta vs Base",
                ],
                isolated_rows or [["None", "n/a", "n/a", "n/a", "n/a"]],
            ),
            "",
            "## Notes",
            "- Variant wall times are not additive; each disabled-block spot check re-runs the pipeline with that single block enabled on top of the base config.",
            "- Block totals do not sum to the full pipeline wall time because the pipeline also includes orchestration, array conversions, logging, object construction, and non-instrumented glue code.",
            "- The `crop` spot check changes the image size from 1920x1536 to 300x300, so its wall-time delta mostly reflects the reduced downstream workload rather than the crop operation itself.",
            "- `yuv_conversion_format` is configured on, but in this profile it bypassed conversion after logging `Invalid input for YUV conversion: RGB image format.`, so no execution-time sample was emitted.",
            "- Results are input-dependent. Different RAW frames, dimensions, algorithms, or hardware backends will change the timing profile.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()

    if args.single_run_config is not None and args.single_run_raw is not None:
        result = run_single_pipeline(
            args.single_run_config,
            args.single_run_raw,
            args.single_run_variant,
        )
        print(f"__PROFILE_RESULT__{json.dumps(result)}")
        return 0

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.raw.exists():
        raise FileNotFoundError(f"RAW file not found: {args.raw}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)

    base_config = load_merged_yaml(pipeline_config_paths(args.config))

    disabled_profile_keys = [
        block["key"]
        for block in PIPELINE_BLOCKS
        if block["key"] in base_config
        and isinstance(base_config[block["key"]], dict)
        and "is_enable" in base_config[block["key"]]
        and not base_config[block["key"]]["is_enable"]
    ]

    variants = [("base", set())] + [(f"{key}_enabled", {key}) for key in disabled_profile_keys]

    all_results: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="isp_profile_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for variant_name, overrides in variants:
            config = build_variant_config(
                base_config,
                log_path=temp_dir / f"{variant_name}.log",
                enabled_overrides=overrides,
            )
            config_path = write_temp_config(config, temp_dir, variant_name)

            for _ in range(args.warmup_runs):
                run_pipeline_once(config_path, args.raw, variant_name)

            measured_runs = [run_pipeline_once(config_path, args.raw, variant_name) for _ in range(args.runs)]
            result = {
                "focus_block": next(iter(overrides), None),
                "runs": measured_runs,
                "summary": summarize_runs(measured_runs),
            }
            all_results[variant_name] = result
            print(
                f"Completed {variant_name}: "
                f"{format_seconds(result['summary']['wall_time_seconds']['mean'])} mean wall time"
            )

    report_text = build_report(
        config_path=args.config,
        raw_path=args.raw,
        runs_per_variant=args.runs,
        warmup_runs=args.warmup_runs,
        base_config=base_config,
        variant_results=all_results,
    )
    args.report.write_text(report_text, encoding="utf-8")
    args.data.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"Report written to {args.report}")
    print(f"Raw profiling data written to {args.data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
