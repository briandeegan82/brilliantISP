#!/usr/bin/env python3
"""
Profile different demosaic algorithms and tone mapping operators.
Focuses on CPU-only profiling for performance comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

from brilliant_isp import BrilliantISP

# Demosaic algorithms to profile (CPU only)
DEMOSAIC_ALGORITHMS = [
    "bilinear",
    "malvar",
    "vng",
    "vng_opt",
    "hamilton_adams",
    "hamilton_adams_opt",
    "ppg",
    "ppg_opt",
    "lmmse",
    "lmmse_opt",
    "lmmse_fast",
    "ahd",
    "ahd_opt",
]

# Tone mapping operators to profile (CPU only)
TONE_MAPPERS = [
    "durand",
    "aces",
    "reinhard_integer",
    "aces_integer",
    "hable",
    "hable_integer",
]


DEFAULT_CONFIG = REPO_ROOT / "config" / "nicos_config.yaml"
DEFAULT_RAW = REPO_ROOT / "in_frames" / "normal" / "avl_image.raw"
DEFAULT_REPORT = REPO_ROOT / "reports" / "demosaic_tmo_profile_report.md"
DEFAULT_DATA = REPO_ROOT / "reports" / "demosaic_tmo_profile_data.json"


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
    parser.add_argument(
        "--profile-demosaic",
        action="store_true",
        default=True,
        help="Profile demosaic algorithms",
    )
    parser.add_argument(
        "--profile-tmo",
        action="store_true",
        default=True,
        help="Profile tone mapping operators",
    )
    parser.add_argument(
        "--demosaic-only",
        action="store_true",
        help="Profile only demosaic algorithms",
    )
    parser.add_argument(
        "--tmo-only",
        action="store_true",
        help="Profile only tone mapping operators",
    )
    return parser.parse_args()


def clear_logger_handlers() -> None:
    """Clear all logger handlers to avoid duplication."""
    logger_names = ["BrilliantISP", "Demosaic", "ToneMapping"]
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def load_and_prepare_config(config_path: Path) -> dict[str, Any]:
    """Load config and disable unnecessary outputs."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Disable all saves and plots
    config["platform"]["disable_progress_bar"] = True
    config["platform"]["plot_histograms"] = False
    config["platform"]["render_3a"] = False
    config["platform"]["debug_enabled"] = True
    config["platform"]["debug_log_level"] = "INFO"

    # Disable GPU operations
    if "gpu" in config.get("platform", {}):
        config["platform"]["gpu"]["enabled"] = False

    # Ensure auto white balance is properly configured
    if config.get("white_balance", {}).get("is_auto"):
        config["auto_white_balance"]["is_enable"] = True

    # Disable all saves
    for key in config:
        if isinstance(config[key], dict) and "is_save" in config[key]:
            config[key]["is_save"] = False

    return config


def profile_demosaic_algorithm(
    config_path: Path,
    raw_path: Path,
    algorithm: str,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Profile a single demosaic algorithm."""
    timings = []

    for run_idx in range(warmup + runs):
        clear_logger_handlers()

        # Load config
        config = load_and_prepare_config(config_path)
        config["demosaic"]["algorithm"] = algorithm

        # Create temp config
        temp_config = config_path.parent / f"temp_demosaic_{algorithm}.yml"
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        try:
            # Run pipeline
            data_path = str(raw_path.parent)
            isp = BrilliantISP(data_path, str(temp_config), outFileName=f"profile_{algorithm}")
            isp.raw_file = raw_path.name

            if isp.c_yaml is None:
                raise RuntimeError("Config failed to load.")

            isp.c_yaml["platform"]["filename"] = raw_path.name

            byte_order = isp.sensor_info["endian_type"] if isp.sensor_info else "ieee-le"
            load_byte_order = "big" if "be" in byte_order else "little"
            isp.load_raw(byte_order=load_byte_order)

            start = time.perf_counter()
            isp.run_pipeline(visualize_output=False)
            elapsed = time.perf_counter() - start

            # Only record after warmup
            if run_idx >= warmup:
                timings.append(elapsed)
                print(f"  {algorithm}: run {run_idx - warmup + 1}/{runs} = {elapsed:.3f}s")

        finally:
            # Cleanup
            if temp_config.exists():
                temp_config.unlink()

    return {
        "algorithm": algorithm,
        "mean": statistics.fmean(timings) if timings else None,
        "stdev": statistics.stdev(timings) if len(timings) > 1 else None,
        "min": min(timings) if timings else None,
        "max": max(timings) if timings else None,
        "timings": timings,
    }


def profile_tone_mapper(
    config_path: Path,
    raw_path: Path,
    tone_mapper: str,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Profile a single tone mapping operator."""
    timings = []

    for run_idx in range(warmup + runs):
        clear_logger_handlers()

        # Load config
        config = load_and_prepare_config(config_path)
        config["tone_mapping"]["tone_mapper"] = tone_mapper
        config["tone_mapping"]["is_enable"] = True

        # Create temp config
        temp_config = config_path.parent / f"temp_tmo_{tone_mapper}.yml"
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        try:
            # Run pipeline
            data_path = str(raw_path.parent)
            isp = BrilliantISP(data_path, str(temp_config), outFileName=f"profile_{tone_mapper}")
            isp.raw_file = raw_path.name

            if isp.c_yaml is None:
                raise RuntimeError("Config failed to load.")

            isp.c_yaml["platform"]["filename"] = raw_path.name

            byte_order = isp.sensor_info["endian_type"] if isp.sensor_info else "ieee-le"
            load_byte_order = "big" if "be" in byte_order else "little"
            isp.load_raw(byte_order=load_byte_order)

            start = time.perf_counter()
            isp.run_pipeline(visualize_output=False)
            elapsed = time.perf_counter() - start

            # Only record after warmup
            if run_idx >= warmup:
                timings.append(elapsed)
                print(f"  {tone_mapper}: run {run_idx - warmup + 1}/{runs} = {elapsed:.3f}s")

        finally:
            # Cleanup
            if temp_config.exists():
                temp_config.unlink()

    return {
        "tone_mapper": tone_mapper,
        "mean": statistics.fmean(timings) if timings else None,
        "stdev": statistics.stdev(timings) if len(timings) > 1 else None,
        "min": min(timings) if timings else None,
        "max": max(timings) if timings else None,
        "timings": timings,
    }


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def format_percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(
    config_path: Path,
    raw_path: Path,
    runs: int,
    warmup: int,
    demosaic_results: dict[str, dict[str, Any]] | None,
    tmo_results: dict[str, dict[str, Any]] | None,
) -> str:
    """Build markdown report from profiling results."""
    lines = [
        "# Demosaic and Tone Mapping Performance Profile",
        "",
        "## Summary",
        "",
        f"- Configuration: `{config_path}`",
        f"- Input image: `{raw_path}`",
        f"- Measured runs per variant: {runs}",
        f"- Warm-up runs per variant: {warmup}",
        f"- Platform: CPU only (GPU disabled)",
        "",
    ]

    # Demosaic results
    if demosaic_results:
        lines.extend(
            [
                "## Demosaic Algorithm Performance",
                "",
            ]
        )

        # Sort by mean time
        sorted_results = sorted(
            demosaic_results.values(),
            key=lambda x: x["mean"] if x["mean"] is not None else float("inf"),
        )

        fastest = sorted_results[0] if sorted_results else None
        slowest = sorted_results[-1] if sorted_results else None

        if fastest:
            lines.append(f"**Fastest**: `{fastest['algorithm']}` at {format_seconds(fastest['mean'])}")
        if slowest:
            lines.append(f"**Slowest**: `{slowest['algorithm']}` at {format_seconds(slowest['mean'])}")
        if fastest and slowest and fastest["mean"] and slowest["mean"]:
            speedup = slowest["mean"] / fastest["mean"]
            lines.append(f"**Speedup**: {speedup:.2f}x (fastest vs slowest)")

        lines.extend(["", "### Detailed Results", ""])

        # Build table
        table_rows = []
        baseline_mean = fastest["mean"] if fastest else None

        for result in sorted_results:
            relative = ""
            if baseline_mean and result["mean"]:
                if result["algorithm"] == fastest["algorithm"]:
                    relative = "baseline"
                else:
                    ratio = result["mean"] / baseline_mean
                    relative = f"{ratio:.2f}x"

            table_rows.append(
                [
                    result["algorithm"],
                    format_seconds(result["mean"]),
                    format_seconds(result["stdev"]),
                    format_seconds(result["min"]),
                    format_seconds(result["max"]),
                    relative,
                ]
            )

        lines.append(
            build_markdown_table(
                ["Algorithm", "Mean", "Stdev", "Min", "Max", "Relative"],
                table_rows,
            )
        )
        lines.extend(["", ""])

    # Tone mapping results
    if tmo_results:
        lines.extend(
            [
                "## Tone Mapping Operator Performance",
                "",
            ]
        )

        # Sort by mean time
        sorted_results = sorted(
            tmo_results.values(),
            key=lambda x: x["mean"] if x["mean"] is not None else float("inf"),
        )

        fastest = sorted_results[0] if sorted_results else None
        slowest = sorted_results[-1] if sorted_results else None

        if fastest:
            lines.append(f"**Fastest**: `{fastest['tone_mapper']}` at {format_seconds(fastest['mean'])}")
        if slowest:
            lines.append(f"**Slowest**: `{slowest['tone_mapper']}` at {format_seconds(slowest['mean'])}")
        if fastest and slowest and fastest["mean"] and slowest["mean"]:
            speedup = slowest["mean"] / fastest["mean"]
            lines.append(f"**Speedup**: {speedup:.2f}x (fastest vs slowest)")

        lines.extend(["", "### Detailed Results", ""])

        # Build table
        table_rows = []
        baseline_mean = fastest["mean"] if fastest else None

        for result in sorted_results:
            relative = ""
            if baseline_mean and result["mean"]:
                if result["tone_mapper"] == fastest["tone_mapper"]:
                    relative = "baseline"
                else:
                    ratio = result["mean"] / baseline_mean
                    relative = f"{ratio:.2f}x"

            table_rows.append(
                [
                    result["tone_mapper"],
                    format_seconds(result["mean"]),
                    format_seconds(result["stdev"]),
                    format_seconds(result["min"]),
                    format_seconds(result["max"]),
                    relative,
                ]
            )

        lines.append(
            build_markdown_table(
                ["Tone Mapper", "Mean", "Stdev", "Min", "Max", "Relative"],
                table_rows,
            )
        )
        lines.extend(["", ""])

    # Notes
    lines.extend(
        [
            "## Notes",
            "",
            "- All measurements are CPU-only (GPU acceleration disabled)",
            "- Times include the entire ISP pipeline execution, not just the profiled module",
            "- Results are input-dependent and may vary with different images",
            "- 'Relative' column shows slowdown factor compared to the fastest variant",
            "- Optimized variants (e.g., `_opt`) typically use vectorized operations for better performance",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not args.raw.exists():
        raise FileNotFoundError(f"RAW file not found: {args.raw}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)

    # Determine what to profile
    profile_demosaic = args.profile_demosaic and not args.tmo_only
    profile_tmo = args.profile_tmo and not args.demosaic_only

    if args.demosaic_only:
        profile_demosaic = True
        profile_tmo = False
    elif args.tmo_only:
        profile_demosaic = False
        profile_tmo = True

    demosaic_results = {}
    tmo_results = {}

    # Profile demosaic algorithms
    if profile_demosaic:
        print(f"\n{'='*70}")
        print("Profiling Demosaic Algorithms (CPU only)")
        print(f"{'='*70}\n")

        for algorithm in DEMOSAIC_ALGORITHMS:
            print(f"\nProfiling demosaic algorithm: {algorithm}")
            try:
                result = profile_demosaic_algorithm(args.config, args.raw, algorithm, args.runs, args.warmup_runs)
                demosaic_results[algorithm] = result
                print(f"  Mean: {format_seconds(result['mean'])}")
            except Exception as e:
                print(f"  ERROR: {e}")
                demosaic_results[algorithm] = {
                    "algorithm": algorithm,
                    "error": str(e),
                    "mean": None,
                    "stdev": None,
                    "min": None,
                    "max": None,
                    "timings": [],
                }

    # Profile tone mapping operators
    if profile_tmo:
        print(f"\n{'='*70}")
        print("Profiling Tone Mapping Operators (CPU only)")
        print(f"{'='*70}\n")

        for tone_mapper in TONE_MAPPERS:
            print(f"\nProfiling tone mapper: {tone_mapper}")
            try:
                result = profile_tone_mapper(args.config, args.raw, tone_mapper, args.runs, args.warmup_runs)
                tmo_results[tone_mapper] = result
                print(f"  Mean: {format_seconds(result['mean'])}")
            except Exception as e:
                print(f"  ERROR: {e}")
                tmo_results[tone_mapper] = {
                    "tone_mapper": tone_mapper,
                    "error": str(e),
                    "mean": None,
                    "stdev": None,
                    "min": None,
                    "max": None,
                    "timings": [],
                }

    # Build report
    report_text = build_report(
        config_path=args.config,
        raw_path=args.raw,
        runs=args.runs,
        warmup=args.warmup_runs,
        demosaic_results=demosaic_results if profile_demosaic else None,
        tmo_results=tmo_results if profile_tmo else None,
    )

    # Save results
    args.report.write_text(report_text, encoding="utf-8")

    all_results = {
        "config": str(args.config),
        "raw_file": str(args.raw),
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "demosaic_results": demosaic_results if profile_demosaic else None,
        "tmo_results": tmo_results if profile_tmo else None,
    }
    args.data.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\n{'='*70}")
    print(f"Report written to: {args.report}")
    print(f"Raw data written to: {args.data}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
