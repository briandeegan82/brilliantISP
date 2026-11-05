from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml

from brilliantisp_gpu.api import process_frame_gpu
from brilliantisp_gpu.config import prepare_config
from brilliantisp_gpu.sources import decode_source
from brilliantisp_gpu.types import GpuOptions
from tools.gpu_validation.contracts import FINAL_FLOAT, FINAL_INTEGER, STATISTICS
from tools.gpu_validation.corpus import case_definitions
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.reference import (
    ROOT, environment, file_identity, gpu_and_harness_sources, jsonable,
    save_array, save_source_files, source_hashes, write_json,
)
from util.config_merge import load_merged_yaml
from util.output_contract import OutputSpec
from util.pipeline_runner import process_frame
from util.pipeline_types import ProcessingOptions


def _write_yaml(path, data):
    with path.open("x") as stream:
        yaml.safe_dump(jsonable(data), stream, sort_keys=False)


def prepare_workloads(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    originals = [
        {"id": "fv", "source": "in_frames/hdr_mode/FV.raw",
         "configs": ["config/SVS_cam.yml"], "overlay": {}},
        {"id": "ad", "source": "in_frames/selected raw files/6ACE750_011_AD_FL_frame_000028.raw",
         "configs": ["config/AD_cam_2.yml"], "overlay": {}},
        *case_definitions(),
    ]
    records = []
    for case in originals:
        directory = output / case["id"]
        directory.mkdir()
        paths = [ROOT / name for name in case["configs"]]
        config_identities = [file_identity(path) for path in paths]
        config = load_merged_yaml(paths)
        for key, value in case["overlay"].items():
            config.setdefault(key, {}).update(value)
        prepared = prepare_config(config, options=ProcessingOptions())
        original = ROOT / case["source"]
        original_identity = file_identity(original)
        decoded = decode_source(
            original, sensor_info=prepared.effective_config["sensor_info"],
            platform=prepared.effective_config["platform"], run_id=case["id"],
        )
        height, width = decoded.pixels.shape
        if min(height, width) < 512:
            raise ValueError(f"source too small for the declared ROI: {original}")
        y, x = ((height - 512) // 4) * 2, ((width - 512) // 4) * 2
        pixels = decoded.pixels[y:y + 512, x:x + 512].copy()
        pixels = pixels.astype(pixels.dtype.newbyteorder("<"), copy=False)
        config = copy.deepcopy(prepared.effective_config)
        sensor = config["sensor_info"]
        sensor.update(
            width=512, height=512, bayer_pattern=decoded.sensor_info["bayer_pattern"],
            source_format="raw", storage_format=pixels.dtype.name,
            data_format=pixels.dtype.name, endian_type="ieee-le", data_alignment="LSB",
            header_bytes=0, row_stride_bytes=None, embedded_rows_top=0,
            embedded_rows_bottom=0, horizontal_flip=False, vertical_flip=False,
        )
        config["crop"]["is_enable"] = False
        config["platform"].update(
            filename=f"{case['id']}.raw", disable_progress_bar=True,
            debug_enabled=False, plot_histograms=False,
        )
        raw, cfg = directory / f"{case['id']}.raw", directory / "config.yml"
        with raw.open("xb") as stream:
            stream.write(pixels.tobytes(order="C"))
        with (directory / "pixels.npy").open("xb") as stream:
            np.save(stream, pixels, allow_pickle=False)
        _write_yaml(cfg, config)
        check = decode_source(
            raw, sensor_info=sensor, platform=config["platform"], run_id=case["id"],
        )
        np.testing.assert_array_equal(check.pixels, pixels)
        result = process_frame(raw, config, options=ProcessingOptions(run_id=case["id"]))
        if original_identity != file_identity(original) or config_identities != [
            file_identity(path) for path in paths
        ]:
            raise RuntimeError("original input/config changed during ROI capture")
        reference = {
            "native": save_array(directory / "cpu_native.npy", result.native_image),
            "preview": save_array(directory / "cpu_preview.npy", result.preview_rgb8),
            "metadata": result.native_metadata, "feedback": result.feedback,
            "effective_config": result.effective_config,
            "migration_report": result.migration_report, "warnings": result.warnings,
        }
        write_json(directory / "cpu_reference.json", reference)
        record = {
            "id": case["id"], "original_source": original_identity,
            "original_configs": config_identities,
            "validation_overlay": case["overlay"], "bbox_yxhw": [y, x, 512, 512],
            "cfa": sensor["bayer_pattern"], "raw": file_identity(raw),
            "config": file_identity(cfg), "pixels": file_identity(directory / "pixels.npy"),
            "cpu_reference": file_identity(directory / "cpu_reference.json"),
            "label": "Real Bayer ROI fixture; CPU dataset extraction is outside every timed region. Not full-frame GPU crop performance.",
        }
        write_json(directory / "dataset.json", record)
        records.append(record)
    write_json(output / "datasets.json", {
        "datasets": records, "environment": environment(),
        "cpu_sources": source_hashes(), "gpu_and_harness_sources": gpu_and_harness_sources(),
    })
    return records


def output_variants():
    return (
        ("png8", dict(format="png", bit_depth=8, transfer="srgb")),
        ("png16", dict(format="png", bit_depth=16, transfer="srgb")),
        ("tiff16", dict(format="tiff", bit_depth=16, transfer="power", gamma=2.2)),
        ("linear_f32", dict(format="tiff", bit_depth=32, transfer="linear")),
        ("jpeg8", dict(format="jpeg", bit_depth=8, transfer="srgb")),
        ("yuv444_8", dict(model="ycbcr", format="yuv", bit_depth=8,
                         transfer="power", gamma=2.2, matrix="bt601",
                         range="limited", layout="444")),
        ("yuv444_16", dict(model="ycbcr", format="yuv", bit_depth=16,
                          transfer="srgb", matrix="bt709", range="full", layout="444")),
        ("yuyv8", dict(model="ycbcr", format="yuv", bit_depth=8,
                      transfer="power", gamma=2.2, matrix="bt601",
                      range="limited", layout="422", packing="yuyv")),
        ("uyvy16", dict(model="ycbcr", format="yuv", bit_depth=16,
                       transfer="power", gamma=2.2, matrix="bt709",
                       range="full", layout="422", packing="uyvy")),
        ("odd_crop_png8", dict(format="png", bit_depth=8, transfer="srgb")),
    )


def _read_artifact(path, spec, shape):
    if spec.format in ("png", "jpeg"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise AssertionError(f"could not decode {path}")
        return image[..., ::-1]
    if spec.format == "tiff":
        return tifffile.imread(path)
    return np.fromfile(path, dtype="<u1" if spec.bit_depth == 8 else "<u2").reshape(shape)


def compare_feedback(expected, actual):
    left, right = jsonable(expected), jsonable(actual)
    fields = {}
    for name in left.keys() | right.keys():
        if name not in left or name not in right:
            fields[name] = {"passed": False, "error": "missing feedback field"}
        elif name in ("awb_gains", "ae_meter_average") and left[name] is not None and right[name] is not None:
            fields[name] = compare_arrays(
                np.atleast_1d(left[name]).astype(np.float64),
                np.atleast_1d(right[name]).astype(np.float64), STATISTICS,
            )
        else:
            fields[name] = {"passed": type(left[name]) is type(right[name])
                            and left[name] == right[name]}
    return {"passed": all(value["passed"] for value in fields.values()), "fields": fields}


def validate_outputs(datasets: Path, output: Path, device: str, *, completed_reports=()):
    implementation, cpu = gpu_and_harness_sources(), source_hashes()
    entries = json.loads((datasets / "datasets.json").read_text())["datasets"]
    expected_keys = {(entry["id"], name) for entry in entries for name, _ in output_variants()}
    completed = {}
    for path in completed_reports:
        path = Path(path).resolve()
        record = json.loads(path.read_text())
        key = (record["dataset"], record["variant"])
        if key not in expected_keys or key in completed or not record["passed"]:
            raise ValueError(f"invalid or duplicate completed output report: {path}")
        old = path.parents[2] / "source"
        critical = {
            key: value for key, value in implementation.items()
            if key.startswith("brilliantisp_gpu/") or key in (
                "isp_pipeline_gpu_batch.py", "tools/gpu_validation/contracts.py",
                "tools/gpu_validation/metrics.py", "tools/gpu_validation/reference.py",
            )
        }
        for kind, identities in (("cpu", cpu), ("gpu_and_harness", critical)):
            if any(file_identity(old / kind / key)["sha256"] != value
                   for key, value in identities.items()):
                raise RuntimeError(f"completed output report uses a different numerical implementation: {path}")
        for artifact in record["artifacts"]:
            if file_identity(Path(artifact["path"])) != artifact:
                raise RuntimeError(f"completed output artifact changed: {artifact['path']}")
        for name in ("gpu_native", "gpu_preview"):
            if file_identity(path.parent / record[name]["file"])["sha256"] != record[name]["sha256"]:
                raise RuntimeError(f"completed frozen array changed: {path} {name}")
        completed[key] = {
            **record, "array_root": str(path.parent),
            "reused_report": file_identity(path),
            "reused_cpu_reference": file_identity(path.parent / "cpu_reference.json"),
        }
    output.mkdir(parents=True, exist_ok=False)
    save_source_files(output / "source" / "gpu_and_harness", implementation)
    save_source_files(output / "source" / "cpu", cpu)
    write_json(output / "run_manifest.json", {
        "datasets": file_identity(datasets / "datasets.json"), "environment": environment(),
        "cpu_sources": cpu, "gpu_and_harness_sources": implementation,
        "completed_reports": [record["reused_report"] for record in completed.values()],
        "output_variants": output_variants(),
    })
    records = []
    for entry in entries:
        source = Path(entry["raw"]["path"])
        for identity in (entry["raw"], entry["config"], entry["pixels"],
                         entry["cpu_reference"], entry["original_source"], *entry["original_configs"]):
            if file_identity(Path(identity["path"])) != identity:
                raise RuntimeError(f"frozen ROI workload input changed: {identity['path']}")
        base = yaml.safe_load(Path(entry["config"]["path"]).read_text())
        for name, section in output_variants():
            if (entry["id"], name) in completed:
                record = completed[(entry["id"], name)]
                if record["metadata"]["source_identity"] != str(source):
                    raise RuntimeError("completed output report uses a different input")
                records.append(record)
                write_json(output / f"partial_results_{len(records):03d}.json", records)
                print(json.dumps({
                    "dataset": entry["id"], "variant": name, "reused_frozen_report": True,
                }), flush=True)
                continue
            directory = output / entry["id"] / name
            directory.mkdir(parents=True)
            config = copy.deepcopy(base)
            config["output"] = section
            if name == "linear_f32":
                config["tone_mapping"]["is_enable"] = False
            if name == "odd_crop_png8":
                config["crop"].update(
                    is_enable=True, crop_x_start=1, crop_y_start=3,
                    new_width=500, new_height=498,
                )
            _write_yaml(directory / "validation_config.yml", config)
            options = ProcessingOptions(
                run_id=f"{entry['id']}_{name}", output_dir=directory / "files",
            )
            expected = process_frame(source, config, options=options)
            write_json(directory / "cpu_reference.json", {
                "native": save_array(directory / "cpu_native.npy", expected.native_image),
                "preview": save_array(directory / "cpu_preview.npy", expected.preview_rgb8),
                "metadata": expected.native_metadata, "feedback": expected.feedback,
                "effective_config": expected.effective_config,
                "migration_report": expected.migration_report,
            })
            actual = process_frame_gpu(
                source, config, options=dataclasses.replace(options, save_outputs=True),
                gpu=GpuOptions(device=device),
            )
            gate = FINAL_FLOAT if expected.native_image.dtype.kind == "f" else FINAL_INTEGER
            spec = OutputSpec.from_mapping(section)
            native = compare_arrays(
                expected.native_image, actual.native_image, gate,
                data_range=1.0 if gate is FINAL_FLOAT else float(2**spec.bit_depth - 1),
            )
            preview = compare_arrays(expected.preview_rgb8, actual.preview_rgb8,
                                     FINAL_INTEGER, data_range=255.0)
            feedback = compare_feedback(expected.feedback, actual.feedback)
            comparisons = {
                "native": native["passed"], "preview": preview["passed"],
                "metadata": expected.native_metadata == actual.native_metadata,
                "feedback": feedback["passed"],
                "effective_config": expected.effective_config == actual.effective_config,
                "migration_report": expected.migration_report == actual.migration_report,
                "warnings": expected.warnings == actual.warnings,
            }
            artifact = actual.artifacts[0]
            decoded = _read_artifact(artifact, spec, actual.native_image.shape)
            if spec.format == "jpeg":
                assert artifact.read_bytes().startswith(b"\xff\xd8")
                assert decoded.dtype == np.uint8 and decoded.shape == actual.native_image.shape
                delta = decoded.astype(np.float64) - actual.native_image.astype(np.float64)
                codec = {"lossy": True, "max_abs": float(np.abs(delta).max()),
                         "mean_abs": float(np.abs(delta).mean()),
                         "rmse": float(np.sqrt(np.mean(delta**2)))}
            else:
                assert decoded.dtype == actual.native_image.dtype
                np.testing.assert_array_equal(decoded, actual.native_image)
                codec = {"lossy": False, "roundtrip_exact": True}
            sidecar = json.loads(artifact.with_suffix(".json").read_text())
            effective = yaml.safe_load(
                artifact.with_name(artifact.stem + "_effective.yaml").read_text())
            migration = json.loads(
                artifact.with_name(artifact.stem + "_migration.json").read_text())
            comparisons["sidecar_metadata"] = sidecar == jsonable(actual.native_metadata)
            comparisons["effective_yaml"] = jsonable(effective) == jsonable(actual.effective_config)
            comparisons["migration_sidecar"] = jsonable(migration) == jsonable(actual.migration_report)
            record = {
                "dataset": entry["id"], "variant": name, "passed": all(comparisons.values()),
                "comparisons": comparisons, "native": native, "preview": preview, "codec": codec,
                "feedback": feedback,
                "gpu_native": save_array(directory / "gpu_native.npy", actual.native_image),
                "gpu_preview": save_array(directory / "gpu_preview.npy", actual.preview_rgb8),
                "artifacts": [file_identity(path) for path in actual.artifacts],
                "native_min": float(actual.native_image.min()),
                "native_max": float(actual.native_image.max()),
                "metadata": actual.native_metadata,
                "array_root": str(directory),
            }
            write_json(directory / "report.json", record)
            records.append(record)
            write_json(output / f"partial_results_{len(records):03d}.json", records)
            print(json.dumps(jsonable({
                "dataset": entry["id"], "variant": name, "passed": record["passed"],
                "comparisons": comparisons,
            })), flush=True)
            if not record["passed"]:
                raise RuntimeError(f"fixed output gate failed: {entry['id']} {name}")
    if cpu != source_hashes() or implementation != gpu_and_harness_sources():
        raise RuntimeError("source changed during output qualification")
    write_json(output / "results.json", records)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "roundtrip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--completed-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_workloads(args.output.resolve())
    elif args.datasets is None:
        parser.error("--datasets is required for roundtrip")
    else:
        validate_outputs(args.datasets.resolve(), args.output.resolve(), args.device,
                         completed_reports=args.completed_report)


if __name__ == "__main__":
    main()
