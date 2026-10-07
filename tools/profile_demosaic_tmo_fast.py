#!/usr/bin/env python3
"""
Fast profiling of demosaic algorithms and tone mapping operators.
Profiles only the specific module execution, not the entire pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")

# Demosaic algorithms to profile (CPU only)
DEMOSAIC_ALGORITHMS = [
    "bilinear",
    "malvar",
    "vng_opt",  # Use optimized version instead of slow VNG
    "hamilton_adams_opt",
    "ppg_opt",
    "lmmse_fast",  # Use fast version
    "ahd_opt",
]

# Tone mapping operators to profile
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
DEFAULT_REPORT = REPO_ROOT / "reports" / "demosaic_tmo_fast_profile_report.md"
DEFAULT_DATA = REPO_ROOT / "reports" / "demosaic_tmo_fast_profile_data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per variant.")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--demosaic-only", action="store_true")
    parser.add_argument("--tmo-only", action="store_true")
    return parser.parse_args()


def profile_demosaic_module(
    raw_bayer: np.ndarray,
    sensor_info: dict,
    platform: dict,
    algorithm: str,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Profile demosaic module directly."""
    from modules.demosaic.demosaic import Demosaic

    timings = []
    demosaic_config = {"is_save": False, "algorithm": algorithm}

    for run_idx in range(warmup + runs):
        # Create fresh demosaic instance for each run
        demosaic = Demosaic(raw_bayer.copy(), platform, sensor_info, demosaic_config)

        start = time.perf_counter()
        try:
            _ = demosaic.execute(algorithm=algorithm)
            elapsed = time.perf_counter() - start

            if run_idx >= warmup:
                timings.append(elapsed)
                print(f"  {algorithm}: run {run_idx - warmup + 1}/{runs} = {elapsed:.3f}s")
        except Exception as e:
            print(f"  {algorithm}: ERROR - {str(e)[:80]}")
            return {
                "algorithm": algorithm,
                "error": str(e),
                "mean": None,
                "stdev": None,
                "min": None,
                "max": None,
            }

    return {
        "algorithm": algorithm,
        "mean": statistics.fmean(timings) if timings else None,
        "stdev": statistics.stdev(timings) if len(timings) > 1 else None,
        "min": min(timings) if timings else None,
        "max": max(timings) if timings else None,
        "timings": timings,
    }


def profile_tone_mapper_module(
    input_img: np.ndarray,
    pipeline_context: Any,
    tone_mapper: str,
    runs: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Profile tone mapping module directly."""
    from modules.tone_mapping.tone_mapping import ToneMapping

    timings = []

    for run_idx in range(warmup + runs):
        # Update tone mapper in context
        pipeline_context.tone_mapping["tone_mapper"] = tone_mapper
        pipeline_context.tone_mapping["is_enable"] = True

        start = time.perf_counter()
        try:
            tmo = ToneMapping(input_img.copy(), pipeline_context)
            _ = tmo.execute()
            elapsed = time.perf_counter() - start

            if run_idx >= warmup:
                timings.append(elapsed)
                print(f"  {tone_mapper}: run {run_idx - warmup + 1}/{runs} = {elapsed:.3f}s")
        except Exception as e:
            print(f"  {tone_mapper}: ERROR - {str(e)[:80]}")
            return {
                "tone_mapper": tone_mapper,
                "error": str(e),
                "mean": None,
                "stdev": None,
                "min": None,
                "max": None,
            }

    return {
        "tone_mapper": tone_mapper,
        "mean": statistics.fmean(timings) if timings else None,
        "stdev": statistics.stdev(timings) if len(timings) > 1 else None,
        "min": min(timings) if timings else None,
        "max": max(timings) if timings else None,
        "timings": timings,
    }


def build_report(
    config_path: Path,
    raw_path: Path,
    runs: int,
    warmup: int,
    demosaic_results: dict[str, dict[str, Any]] | None,
    tmo_results: dict[str, dict[str, Any]] | None,
) -> str:
    """Build markdown report."""
    lines = [
        "# Demosaic and Tone Mapping Performance Profile (Fast Mode)",
        "",
        "## Summary",
        "",
        f"- Configuration: `{config_path}`",
        f"- Input image: `{raw_path}`",
        f"- Measured runs per variant: {runs}",
        f"- Warm-up runs per variant: {warmup}",
        f"- Mode: Direct module profiling (not full pipeline)",
        f"- Platform: CPU only (GPU disabled)",
        "",
    ]

    def format_time(t: float | None) -> str:
        return f"{t:.3f}s" if t is not None else "n/a"

    def build_table(headers: list[str], rows: list[list[str]]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    if demosaic_results:
        lines.extend(["## Demosaic Algorithm Performance", ""])

        sorted_results = sorted(
            [r for r in demosaic_results.values() if r.get("mean") is not None],
            key=lambda x: x["mean"],
        )

        if sorted_results:
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            lines.append(f"**Fastest**: `{fastest['algorithm']}` at {format_time(fastest['mean'])}")
            lines.append(f"**Slowest**: `{slowest['algorithm']}` at {format_time(slowest['mean'])}")
            if fastest["mean"] and slowest["mean"]:
                speedup = slowest["mean"] / fastest["mean"]
                lines.append(f"**Speedup**: {speedup:.2f}x (fastest vs slowest)")
            lines.extend(["", "### Detailed Results", ""])

            table_rows = []
            baseline_mean = fastest["mean"]

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
                        format_time(result["mean"]),
                        format_time(result.get("stdev")),
                        format_time(result.get("min")),
                        format_time(result.get("max")),
                        relative,
                    ]
                )

            lines.append(
                build_table(
                    ["Algorithm", "Mean", "Stdev", "Min", "Max", "Relative"],
                    table_rows,
                )
            )
            lines.extend(["", ""])

    if tmo_results:
        lines.extend(["## Tone Mapping Operator Performance", ""])

        sorted_results = sorted(
            [r for r in tmo_results.values() if r.get("mean") is not None],
            key=lambda x: x["mean"],
        )

        if sorted_results:
            fastest = sorted_results[0]
            slowest = sorted_results[-1]
            lines.append(f"**Fastest**: `{fastest['tone_mapper']}` at {format_time(fastest['mean'])}")
            lines.append(f"**Slowest**: `{slowest['tone_mapper']}` at {format_time(slowest['mean'])}")
            if fastest["mean"] and slowest["mean"]:
                speedup = slowest["mean"] / fastest["mean"]
                lines.append(f"**Speedup**: {speedup:.2f}x (fastest vs slowest)")
            lines.extend(["", "### Detailed Results", ""])

            table_rows = []
            baseline_mean = fastest["mean"]

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
                        format_time(result["mean"]),
                        format_time(result.get("stdev")),
                        format_time(result.get("min")),
                        format_time(result.get("max")),
                        relative,
                    ]
                )

            lines.append(
                build_table(
                    ["Tone Mapper", "Mean", "Stdev", "Min", "Max", "Relative"],
                    table_rows,
                )
            )
            lines.extend(["", ""])

    lines.extend(
        [
            "## Notes",
            "",
            "- Fast mode: Profiles only the demosaic/tone mapping module execution",
            "- All measurements are CPU-only (GPU acceleration disabled)",
            "- Results exclude pipeline overhead (loading, white balance, CCM, etc.)",
            "- Using optimized and fast variants where available",
            "- Results are input-dependent and may vary with different images",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.raw.exists():
        raise FileNotFoundError(f"RAW file not found: {args.raw}")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.data.parent.mkdir(parents=True, exist_ok=True)

    # Load config and data
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from brilliant_isp import BrilliantISP

    print(f"\n{'='*70}")
    print("Loading ISP and preparing data...")
    print(f"{'='*70}\n")

    # Load ISP to get preprocessed data
    data_path = str(args.raw.parent)
    isp = BrilliantISP(data_path, str(args.config), outFileName="profile_temp")
    isp.raw_file = args.raw.name
    isp.c_yaml["platform"]["filename"] = args.raw.name

    byte_order = isp.sensor_info["endian_type"] if isp.sensor_info else "ieee-le"
    load_byte_order = "big" if "be" in byte_order else "little"
    isp.load_raw(byte_order=load_byte_order)

    # Run up to white balance to get preprocessed data
    isp.run_pipeline_up_to_wb()

    # Get preprocessed raw for demosaic
    raw_bayer = isp.current_frame
    sensor_info = isp.sensor_info
    platform = isp.platform

    profile_demosaic = not args.tmo_only
    profile_tmo = not args.demosaic_only

    demosaic_results = {}
    tmo_results = {}

    # Profile demosaic
    if profile_demosaic:
        print(f"\n{'='*70}")
        print("Profiling Demosaic Algorithms (CPU only)")
        print(f"{'='*70}\n")

        for algorithm in DEMOSAIC_ALGORITHMS:
            print(f"\nProfiling: {algorithm}")
            result = profile_demosaic_module(raw_bayer, sensor_info, platform, algorithm, args.runs, args.warmup_runs)
            demosaic_results[algorithm] = result
            if result.get("mean"):
                print(f"  Mean: {result['mean']:.3f}s")

    # Profile tone mapping
    if profile_tmo:
        print(f"\n{'='*70}")
        print("Profiling Tone Mapping Operators (CPU only)")
        print(f"{'='*70}\n")

        # Get input for tone mapping (after white balance, before demosaic)
        tmo_input = raw_bayer.copy()

        for tone_mapper in TONE_MAPPERS:
            print(f"\nProfiling: {tone_mapper}")
            result = profile_tone_mapper_module(tmo_input, isp, tone_mapper, args.runs, args.warmup_runs)
            tmo_results[tone_mapper] = result
            if result.get("mean"):
                print(f"  Mean: {result['mean']:.3f}s")

    # Build and save report
    report_text = build_report(
        args.config,
        args.raw,
        args.runs,
        args.warmup_runs,
        demosaic_results if profile_demosaic else None,
        tmo_results if profile_tmo else None,
    )

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
    print(f"Report: {args.report}")
    print(f"Data: {args.data}")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
