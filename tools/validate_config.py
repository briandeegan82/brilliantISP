"""Standalone pipeline-configuration validator.

Usage (from the repo root or any working directory):

    python tools/validate_config.py --config config/SVS_cam.yml
    python tools/validate_config.py --config config/SVS_cam.yml \
        --config config/experiment.yml --json

``--config`` is repeatable; paths are resolved through the same
``pipeline_config_paths`` rules as the pipeline (bare names under ``config/``,
``*_cam.yml`` after the implicit base when present) and merged in argument
order — later configs override earlier ones. ``extends`` inheritance is
honored. ``schema_version: 2`` sources are validated against the native
required-key set (``output`` section, no retired color sections).

Exit codes: 0 = valid (warnings allowed), 1 = validation errors,
2 = config could not be loaded/parsed/normalized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from util.config_merge import load_merged_yaml, pipeline_config_paths
from util.config_schema import (
    NATIVE_REQUIRED_CONFIG_KEYS,
    REQUIRED_CONFIG_KEYS,
    ValidationIssue,
    normalize_config,
    validate_config,
)

EXPECTED_ERRORS = (ValueError, TypeError, OSError, yaml.YAMLError)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_config",
        description=(
            "Validate BrilliantISP pipeline configuration files without "
            "running the pipeline. Repeatable --config merges in argument "
            "order (later files override earlier ones); v2 sources use "
            "the native required-key set."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="PATH",
        help="YAML config path; repeat to overlay multiple files in order",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the result as a single JSON object on stdout",
    )
    return parser


def _resolve_sources(config_args: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for arg in config_args:
        resolved.extend(pipeline_config_paths(arg))
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sources = _resolve_sources(args.config)
    except EXPECTED_ERRORS as exc:
        return _report_load_error(args.json, [], exc)

    try:
        merged = load_merged_yaml(sources)
        normalized, notices = normalize_config(merged)
    except EXPECTED_ERRORS as exc:
        return _report_load_error(args.json, sources, exc)

    version = normalized.get("schema_version")
    required = (
        NATIVE_REQUIRED_CONFIG_KEYS
        if isinstance(version, int) and not isinstance(version, bool) and version == 2
        else REQUIRED_CONFIG_KEYS
    )
    issues = validate_config(normalized)
    for key in required:
        if key not in normalized:
            issues.append(
                ValidationIssue(key, f"missing required section '{key}'")
            )
    errors = [i for i in issues if i.severity == "error"]
    valid = not errors

    if args.json:
        print(
            json.dumps(
                {
                    "valid": valid,
                    "config_sources": [str(p) for p in sources],
                    "notices": notices,
                    "issues": [
                        {"path": i.path, "message": i.message, "severity": i.severity}
                        for i in issues
                    ],
                },
                indent=2,
            )
        )
    else:
        print("Configuration sources:")
        for p in sources:
            print(f"  {p}")
        for notice in notices:
            print(f"notice: {notice}")
        for issue in issues:
            print(f"{issue.severity}: {issue.path}: {issue.message}")
        if valid:
            print("Configuration valid")

    return 0 if valid else 1


def _report_load_error(as_json: bool, sources: list[Path], exc: Exception) -> int:
    if as_json:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "config_sources": [str(p) for p in sources],
                },
                indent=2,
            )
        )
    else:
        print(
            f"error: could not load configuration: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return 2


if __name__ == "__main__":
    sys.exit(main())
