from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import yaml

from tools.gpu_validation.pipeline import validate_reference
from tools.gpu_validation.reference import (
    ROOT, capture_reference, environment, file_identity, gpu_and_harness_sources,
    jsonable, save_source_files, source_hashes, write_json,
)


def case_definitions():
    return (
        {
            "id": "sdr_colorchecker",
            "source": "in_frames/normal/ColorCheckerRaw_100DPs_ISO100_2592x1536_12bits_RGGB.raw",
            "configs": ["config/ROD_cam_daytime_le.yml"],
            "overlay": {},
            "notes": "Unmodified matching 2592x1536 RGGB12 little-endian camera preset.",
        },
        {
            "id": "triton_sdr_tiff",
            "source": "in_frames/hdr_mode/lauren_triton/TRI054S-C_221400748__260324154604_image6.tiff",
            "configs": ["config/triton_lauren_cam.yml"],
            "overlay": {"sensor_info": {"source_format": "tiff", "storage_format": "uint16"}},
            "notes": "TIFF is YX uint16, description declares BayerPattern RGGB and original BayerRG12p. This qualifies the unpacked TIFF, not the unsupported packed RAW.",
        },
        {
            "id": "triton_hdr24_overlay",
            "source": "in_frames/hdr_mode/lauren_triton/TRI054S-C_221400748__260324153544_image4.raw",
            "configs": ["config/triton_lauren_cam.yml", "config/triton_lauren_hdr24_cam.yml"],
            "overlay": {"sensor_info": {
                "source_format": "raw", "storage_format": "packed24",
                "data_alignment": "LSB", "header_bytes": 0, "row_stride_bytes": None,
                "embedded_rows_top": 0, "embedded_rows_bottom": 0,
                "horizontal_flip": False, "vertical_flip": False,
            }},
            "notes": "Explicit validation composition, not the unmodified incomplete HDR preset: same-camera complete SDR preset supplies missing sections, HDR24 overlay supplies its controls. The HDR preset documents 3-byte little-endian BayerRG24; storage_format makes that contract explicit. Missing base_hdr.yml remains a preset limitation.",
        },
    )


def _copy_arrays(source, target, records):
    for record in records:
        path = target / record["file"]
        if not path.exists():
            shutil.copyfile(source / record["file"], path)


def run_corpus(output: Path, scratch: Path, device: str, *, case_ids=None):
    definitions = case_definitions()
    if case_ids is not None:
        wanted = set(case_ids)
        unknown = wanted - {case["id"] for case in definitions}
        if unknown:
            raise ValueError(f"unknown corpus case ids: {sorted(unknown)}")
        definitions = tuple(case for case in definitions if case["id"] in wanted)
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    implementation = gpu_and_harness_sources()
    save_source_files(output / "source" / "cpu", before)
    save_source_files(output / "source" / "gpu_and_harness", implementation)
    git = {
        name: subprocess.run(
            ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
        for name, args in (
            ("head", ("rev-parse", "HEAD")),
            ("status", ("status", "--porcelain=v1", "--untracked-files=normal")),
        )
    }
    write_json(output / "run_manifest.json", {
        "schema_version": 1, "environment": environment(), "git": git,
        "cpu_source_hashes": before, "implementation_hashes": implementation,
        "numeric_contract": file_identity(ROOT / "tools/gpu_validation/contracts.py"),
        "cases": definitions, "device": device, "scratch": str(scratch),
        "storage_policy": "Large per-stage CPU/GPU arrays are in tmpfs. Durable reference manifests, final native/preview arrays, complete comparison metrics, configs, and source snapshots are under this output directory.",
        "blocked": [
            {"case": "triton_packed12_raw", "reason": "Shared CPU decoder does not support BayerRG12p. TIFF counterpart is tested separately."},
            {"case": "unmodified_triton_hdr24_preset", "reason": "Missing config/base_hdr.yml and implicit uint32 declaration for a documented 3-byte RAW. Explicit recorded composition tested separately."},
            {"case": "rocm_hardware", "reason": "No supported ROCm runtime/device is provisioned on this host."},
        ],
    })
    results = []
    for case in definitions:
        case_root = output / case["id"]
        case_root.mkdir()
        case_scratch = scratch / case["id"]
        case_scratch.mkdir()
        configs = [ROOT / path for path in case["configs"]]
        if case["overlay"]:
            overlay = case_root / "validation_overlay.yml"
            with overlay.open("x") as stream:
                yaml.safe_dump(case["overlay"], stream, sort_keys=False)
            configs.append(overlay.resolve())
        write_json(case_root / "case.json", {
            **case, "source_identity": file_identity(ROOT / case["source"]),
            "config_chain": [file_identity(p) for p in configs],
        })
        print(f"CAPTURE {case['id']}", flush=True)
        try:
            reference = case_scratch / "reference"
            frozen = capture_reference(ROOT / case["source"], configs, reference)
            durable_ref = case_root / "reference"
            durable_ref.mkdir()
            shutil.copyfile(reference / "manifest.json", durable_ref / "manifest.json")
            _copy_arrays(reference, durable_ref, (frozen["native"], frozen["preview"]))
            print(f"VALIDATE {case['id']}", flush=True)
            gpu_root = case_scratch / "gpu"
            report = validate_reference(reference, gpu_root, device)
            _copy_arrays(gpu_root, case_root, report["arrays"].values())
            report["storage"] = {
                "stage_array_root": str(gpu_root), "reference_array_root": str(reference),
                "final_array_root": str(case_root.resolve()),
                "durable_reference_manifest": str((durable_ref / "manifest.json").resolve()),
            }
            write_json(case_root / "report.json", report)
            results.append({
                "id": case["id"], "passed": report["passed"],
                "native": report["comparisons"]["native"],
                "preview": report["comparisons"]["preview"],
                "report": str((case_root / "report.json").resolve()),
            })
            print(json.dumps(jsonable(results[-1]), allow_nan=False), flush=True)
            if not report["passed"]:
                raise RuntimeError(f"fixed numerical or metadata gate failed: {case['id']}")
        except BaseException as exc:
            write_json(case_root / "failure.json", {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(), "scratch": str(case_scratch),
            })
            write_json(output / "partial_results.json", results)
            raise
    if source_hashes() != before or gpu_and_harness_sources() != implementation:
        raise RuntimeError("implementation changed during corpus qualification")
    write_json(output / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case", action="append", choices=[
        case["id"] for case in case_definitions()
    ])
    args = parser.parse_args()
    run_corpus(args.output.resolve(), args.scratch.resolve(), args.device,
               case_ids=args.case)


if __name__ == "__main__":
    main()
