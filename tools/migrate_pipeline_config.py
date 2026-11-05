#!/usr/bin/env python3
"""Migrate brilliantISP v1 pipeline configs to native schema_version 2.

Reads source YAMLs (repeatable --config merges in order; 'extends'
inheritance is resolved by the config loader), runs the settled
migrate_pipeline_config canonicalization, validates the migrated mapping
(both validator issues and the native v2 required sections), and writes
exclusive new outputs — never overwrites existing files or symlinks.
Source files are never touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

# Allow running as a bare script: tools/migrate_pipeline_config.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.config_merge import load_merged_yaml, pipeline_config_paths
from util.config_schema import (
    NATIVE_REQUIRED_CONFIG_KEYS,
    normalize_config,
    validate_config,
)
from util.pipeline_migration import migrate_pipeline_config

EXPECTED_ERRORS = (ValueError, TypeError, OSError, yaml.YAMLError)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migrate_pipeline_config",
        description=(
            "Migrate a v1 BrilliantISP config to the native schema_version 2 "
            "mapping. Repeatable --config overlays in argument order; "
            "'extends:' inheritance is resolved first. Both output paths "
            "must be new files (no overwrite, no symlink following)."
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
        "--output",
        required=True,
        type=Path,
        metavar="PATH",
        help="exclusive output YAML path (must not already exist)",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        metavar="PATH",
        help="exclusive migration-report JSON path (must not already exist)",
    )
    return parser


def _exclusive_target(path: Path) -> Path:
    """Validate a target path: parent exists, path is new, not a symlink."""
    if path.is_symlink():
        raise ValueError(f"output already exists, refusing to overwrite symlink: {path}")
    resolved = path.resolve()
    if not resolved.parent.is_dir():
        raise ValueError(f"output directory does not exist: {resolved.parent}")
    if resolved.exists() or resolved.is_symlink():
        raise ValueError(f"output already exists, refusing to overwrite: {resolved}")
    return resolved


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_migrate(config_args: list[str], output: Path, report: Path) -> tuple[Path, Path]:
    sources: list[Path] = []
    for arg in config_args:
        sources.extend(pipeline_config_paths(arg))
    merged = load_merged_yaml(sources)

    migrated, report_data = migrate_pipeline_config(merged, canonicalize_tone_mapping=True)
    normalized, _notices = normalize_config(migrated)
    issues = [i for i in validate_config(normalized) if i.severity == "error"]
    if issues:
        raise ValueError(
            "migrated config failed validation: "
            + "; ".join(f"{i.path}: {i.message}" for i in issues))
    missing = [k for k in NATIVE_REQUIRED_CONFIG_KEYS if k not in normalized]
    if missing:
        raise ValueError(
            f"migrated config missing native required sections: {missing}")

    out_path = _exclusive_target(Path(output))
    report_path = _exclusive_target(Path(report))
    if out_path == report_path:
        raise ValueError("output and report paths must be distinct")

    # Serialize both payloads before touching the filesystem.
    out_text = yaml.safe_dump(dict(normalized), sort_keys=False)
    report_text = json.dumps(_jsonable(report_data), indent=2, sort_keys=True)

    created: list[Path] = []
    try:
        with ExitStack() as stack:
            out_fh = stack.enter_context(out_path.open("x", encoding="utf-8"))
            created.append(out_path)
            rep_fh = stack.enter_context(report_path.open("x", encoding="utf-8"))
            created.append(report_path)
            out_fh.write(out_text)
            rep_fh.write(report_text)
    except Exception:
        for p in created:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return out_path, report_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output, report = run_migrate(list(args.config), args.output, args.report)
    except EXPECTED_ERRORS as exc:
        print(f"migrate_pipeline_config: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote migrated config: {output}")
    print(f"Wrote migration report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
