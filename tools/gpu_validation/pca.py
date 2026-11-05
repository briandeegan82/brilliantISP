from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.gpu_validation.contracts import EXACT, FLOAT64_HELPER, STATISTICS
from tools.gpu_validation.metrics import compare_arrays
from tools.gpu_validation.reference import (
    file_identity, gpu_and_harness_sources, jsonable, save_array, save_source_files,
    source_hashes, write_json,
)


def load_array(directory: Path, record: dict[str, Any]) -> np.ndarray:
    path = directory / record["file"]
    if file_identity(path)["sha256"] != record["sha256"]:
        raise ValueError(f"reference archive identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = archive["pixels"].copy()
    if list(result.shape) != record["shape"] or result.dtype.str != record["dtype"]:
        raise ValueError(f"reference array contract mismatch: {path}")
    return result


def freeze_real_case(reference: Path, output: Path) -> None:
    from modules.auto_white_balance.auto_white_balance import AutoWhiteBalance
    from util.signal_range import linear_code_range

    manifest_path = reference / "manifest.json"
    frozen = json.loads(manifest_path.read_text())
    if frozen["source_hashes"] != source_hashes():
        raise RuntimeError("CPU source identity no longer matches the frozen image reference")
    config = frozen["effective_config"]
    assert config["auto_white_balance"]["algorithm"] == "pca"
    bnr = [s for s in frozen["stages"] if s["stage_id"] == "bayer_noise_reduction"][-1]
    pass_key = (bnr["pass_index"], bnr["ae_pass"])
    same_pass = {
        s["stage_id"]: s for s in frozen["stages"]
        if (s["pass_index"], s["ae_pass"]) == pass_key
    }
    index = same_pass["digital_gain"]["members"][1]["value"]
    applied_gain = (
        float(config["digital_gain"]["gain_array"][index])
        if config["digital_gain"].get("is_enable", True) else 1.0
    )
    sensor = dict(config["sensor_info"])
    sensor["hdr_bit_depth"] = linear_code_range(config).bit_depth
    assert float(2**sensor["hdr_bit_depth"] - 1) == frozen["feedback"]["linear_reference_code_max"]
    codes = load_array(reference, bnr["array"])
    assert codes.shape == (sensor["height"], sensor["width"])
    awb = AutoWhiteBalance(
        codes, sensor, config["auto_white_balance"], config["white_balance"]
    )
    rows = awb._awb_rows(input_gain=applied_gain)
    if rows.shape[0] != frozen["feedback"]["awb_sample_count"]:
        raise AssertionError("reconstructed CPU sample count differs from frozen feedback")
    percentage = float(config["auto_white_balance"]["percentage"])
    means = rows.mean(axis=0)
    unit = means / np.linalg.norm(means)
    projections = rows @ unit
    order = np.argsort(projections)
    count = max(1, int(np.ceil(rows.shape[0] * percentage / 100)))
    selected = np.concatenate((order[:count], order[-count:]))
    covariance = rows[selected].T @ rows[selected]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvector = np.abs(eigenvectors[:, int(np.argmax(eigenvalues))])
    gains = np.clip((eigenvector[1] / eigenvector[[0, 2]]), 0.25, 4.0)
    np.testing.assert_allclose(gains, frozen["feedback"]["awb_gains"],
                               atol=STATISTICS.atol, rtol=STATISTICS.rtol)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "rows": rows, "means": means, "unit": unit, "projections": projections,
        "order": order, "selected_indices": selected, "covariance": covariance,
        "eigenvalues": eigenvalues, "eigenvector": eigenvector, "gains": gains,
    }
    record = {
        "schema_version": 1, "source_reference": file_identity(manifest_path),
        "source_hashes": frozen["source_hashes"], "stage": bnr,
        "percentage": percentage, "sample_count": len(rows), "input_gain": applied_gain,
        "reference_sort": "numpy-1.26.4-scalar-introsort", "numpy": np.__version__,
        "harness": file_identity(Path(__file__)),
        "arrays": {name: save_array(output / f"{name}.npy", a) for name, a in arrays.items()},
    }
    write_json(output / "manifest.json", record)


def validate_case(reference: Path, output: Path, device: str) -> dict[str, Any]:
    import torch

    from brilliantisp_gpu.blocks.statistics import pca_means
    from brilliantisp_gpu.device import gpu_scope, device_properties, resolve_device

    frozen = json.loads((reference / "manifest.json").read_text())
    if frozen["source_hashes"] != source_hashes():
        raise RuntimeError("CPU reference identity changed")
    implementations = gpu_and_harness_sources()
    arrays = {name: load_array(reference, a) for name, a in frozen["arrays"].items()}
    output.mkdir(parents=True, exist_ok=False)
    dev = resolve_device(device)
    with gpu_scope(dev):
        rows = torch.as_tensor(arrays["rows"], device=dev, dtype=torch.float64)
        result, diagnostics = pca_means(
            rows, frozen["percentage"], return_diagnostics=True
        )
        members = (
            {f.name: getattr(diagnostics, f.name) for f in dataclasses.fields(diagnostics)}
            if dataclasses.is_dataclass(diagnostics) else dict(diagnostics)
        )
        actual = {}
        for name in (
            "means", "unit", "projections", "order", "selected_indices",
            "covariance", "eigenvalues", "eigenvector",
        ):
            tensor = members[name]
            if not isinstance(tensor, torch.Tensor) or tensor.device != dev:
                raise AssertionError(f"PCA diagnostic is not resident on selected GPU: {name}")
            actual[name] = tensor.detach().cpu().numpy()
        if result.device != dev:
            raise AssertionError("PCA result is not GPU-resident")
        actual["eigenvector"] = result.detach().cpu().numpy()
        actual["gains"] = torch.clamp(
            result[1] / result[[0, 2]], 0.25, 4.0
        ).detach().cpu().numpy()
    comparisons = {}
    for name, value in actual.items():
        gate = EXACT if name in ("order", "selected_indices") else (
            STATISTICS if name in ("eigenvector", "gains") else FLOAT64_HELPER
        )
        comparisons[name] = compare_arrays(arrays[name], value, gate)
    saved = {name: save_array(output / f"{name}.npy", a) for name, a in actual.items()}
    if implementations != gpu_and_harness_sources():
        raise RuntimeError("GPU or validation implementation changed during measurement")
    save_source_files(output / "source", implementations)
    record = {
        "schema_version": 1, "passed": all(v["passed"] for v in comparisons.values()),
        "source_reference": file_identity(reference / "manifest.json"),
        "gpu": device_properties(dev), "harness": file_identity(Path(__file__)),
        "comparisons": comparisons, "arrays": saved, "implementation_hashes": implementations,
    }
    write_json(output / "report.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "validate"))
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_real_case(args.reference, args.output)
        print(args.output / "manifest.json")
    else:
        report = validate_case(args.reference, args.output, args.device)
        print(json.dumps(jsonable({"passed": report["passed"], "comparisons": report["comparisons"]})))
        raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
