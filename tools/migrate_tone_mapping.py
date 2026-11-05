"""CLI: migrate legacy tone-mapping config to canonical (schema v1).

``--config`` repeatable (merged in order); ``--output`` and ``--report``
are required NEW files (existing files are never overwritten, parent
directories must exist, sources untouched).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
from pathlib import Path

import yaml

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.tone_mapping.migration import migrate_tone_mapping
from util.config_merge import load_merged_yaml, pipeline_config_paths
from util.config_schema import REQUIRED_CONFIG_KEYS, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True,
                        help="YAML config (repeatable, merged in order)")
    parser.add_argument("--output", required=True,
                        help="new YAML path for the migrated config")
    parser.add_argument("--report", required=True,
                        help="new JSON path for the migration report")
    args = parser.parse_args(argv)

    output = Path(args.output)
    report_path = Path(args.report)
    for path, label in ((output, "--output"), (report_path, "--report")):
        if path.exists():
            print(f"error: {label} {path} already exists; refusing to overwrite", file=sys.stderr)
            return 2
        if not path.parent.is_dir():
            print(f"error: {label} parent {path.parent} does not exist", file=sys.stderr)
            return 2
    if output == report_path:
        print("error: --output and --report must be different files", file=sys.stderr)
        return 2

    created: list[Path] = []
    try:
        paths: list[Path] = []
        for raw in args.config:
            paths.extend(pipeline_config_paths(raw))
        merged = load_merged_yaml(paths)
        migrated, report = migrate_tone_mapping(merged)
        errors = [i for i in validate_config(migrated) if i.severity == "error"]
        missing = [key for key in REQUIRED_CONFIG_KEYS if key not in migrated]
        if missing:
            raise ValueError(f"migrated config is missing required sections: {missing}")
        if errors:
            raise ValueError(
                "migrated config failed validation: "
                + "; ".join(f"{i.path}: {i.message}" for i in errors)
            )
        with ExitStack() as stack:
            config_file = stack.enter_context(output.open("x", encoding="utf-8"))
            created.append(output)
            report_file = stack.enter_context(report_path.open("x", encoding="utf-8"))
            created.append(report_path)
            yaml.safe_dump(migrated, config_file, sort_keys=False)
            json.dump(report, report_file, indent=2)
    except FileNotFoundError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        for path in created:
            path.unlink(missing_ok=True)
        print(f"error: migration failed: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {output} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
