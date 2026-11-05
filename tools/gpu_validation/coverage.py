from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from tools.gpu_validation.contracts import DEMOSAIC_ALGORITHMS
from tools.gpu_validation.reference import ROOT, file_identity, gpu_and_harness_sources, source_hashes
from util.config_merge import load_merged_yaml


def _case(identifier, source, configs, *, overlay=None, evidence="", extra_inputs=()):
    paths = [ROOT / name for name in configs]
    config = load_merged_yaml(paths)
    overlay = {} if overlay is None else copy.deepcopy(overlay)
    for name, values in overlay.items():
        config.setdefault(name, {}).update(values)
    return {
        "id": identifier, "source": str(ROOT / source), "config": config,
        "validation_overlay": overlay, "assignment_evidence": evidence,
        "input_identities": [
            file_identity(ROOT / source), *[file_identity(path) for path in paths],
            *[file_identity(ROOT / path) for path in extra_inputs],
        ],
    }


def _verified_report(path):
    path = ROOT / path
    record = json.loads(path.read_text())
    if not record["passed"] or not all(value["passed"] for value in record["comparisons"].values()):
        raise RuntimeError(f"not a qualified frozen report: {path}")
    current = gpu_and_harness_sources()
    runtime = {key: value for key, value in current.items() if key.startswith("brilliantisp_gpu/")}
    if runtime != {
        key: value for key, value in record["implementation_hashes"].items()
        if key.startswith("brilliantisp_gpu/")
    }:
        raise RuntimeError(f"frozen report used a different GPU implementation: {path}")
    manifest = Path(record["source_reference"]["path"])
    frozen = json.loads(manifest.read_text())
    if file_identity(manifest) != record["source_reference"] or frozen["source_hashes"] != source_hashes():
        raise RuntimeError(f"frozen reference identity changed: {manifest}")
    if file_identity(Path(frozen["source"]["path"])) != frozen["source"]:
        raise RuntimeError("frozen input identity changed")
    return {"status": "qualified_existing", "report": file_identity(path),
            "source": frozen["source"], "native": record["comparisons"]["native"],
            "preview": record["comparisons"]["preview"]}


def corpus_definitions():
    inventory = json.loads((ROOT / "tmp/gpu_validation/m0_reference/raw_inventory.json").read_text())
    existing = (
        "tmp/gpu_validation/m5/fv_ad_post_bnr_28b557d5/fv/report.json",
        "tmp/gpu_validation/m5/fv_ad_post_bnr_28b557d5/ad/report.json",
        "tmp/gpu_validation/m5/sdr_exp_fix_de876dc9/report.json",
        "tmp/gpu_validation/m5/triton_qualification_17739e72/triton_sdr_tiff/report.json",
        "tmp/gpu_validation/m5/triton_qualification_17739e72/triton_hdr24_overlay/report.json",
    )
    reused = [_verified_report(path) for path in existing]
    qualified_sources = {row["source"]["path"] for row in reused}
    ready, blocked = [], []
    for row in inventory["cases"]:
        source = Path(row["path"])
        actual = file_identity(source)
        if actual["sha256"] != row["sha256"] or actual["bytes"] != row["bytes"]:
            raise RuntimeError(f"original inventory input changed: {source}")
        relative = row["relative_path"]
        if str(source) in qualified_sources:
            continue
        sidecar = source.with_suffix(".raw.json")
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata["pixelformat"] == "BayerRG12p":
                blocked.append({
                    "status": "blocked", "source": actual,
                    "metadata": file_identity(sidecar),
                    "reason": "Packed BayerRG12p is unsupported by the shared CPU/GPU RAW decoder; do not reinterpret as uint16. Corresponding unpacked TIFF is evaluated separately.",
                })
                continue
            if metadata["pixelformat"] == "BayerRG24" and source.name.endswith("image10.raw"):
                ready.append(_case(
                    "updated_hdr24_raw", relative,
                    ["in_frames/updated_images/configs/imx490_hdr_24.yml"],
                    overlay={"sensor_info": {
                        "source_format": "raw", "storage_format": "packed24",
                        "data_alignment": "LSB", "endian_type": "ieee-le",
                    }},
                    evidence="Preset filename matches image10; sidecar declares BayerRG24/2880x1860. Original uint32 storage does not match 3-byte packing, and endian alias little is invalid under the shared schema: recorded packed24/ieee-le overlay preserves the declared byte order, not unmodified-preset qualification.",
                    extra_inputs=[str(sidecar.relative_to(ROOT))],
                ))
                continue
            raise ValueError(f"unassigned sidecar format: {sidecar}")
        if "_AD_FL_" in source.name:
            ready.append(_case(
                "ad_" + source.stem, relative, ["config/AD_cam_2.yml"],
                evidence="Explicit AD_FL camera-family tag matched to sensor AD preset and the frozen matched-series frame000028 configuration. Encoding/CFA remain the preset declarations, not independently inferred from file size.",
            ))
        elif source.name.startswith("SVS_"):
            ready.append(_case(
                source.stem.lower(), relative, ["config/SVS_cam.yml"],
                evidence="Explicit SVS camera-family filename matched to sensor SVS preset and frozen FV/SVS configuration; encoding/CFA are the declared preset profile.",
            ))
        elif source.name.startswith("ColorChecker"):
            ready.append(_case(
                "sdr_" + source.stem, relative, ["config/ROD_cam_daytime_le.yml"],
                evidence="Input filename explicitly declares RGGB/12-bit/2592x1536; the existing ROD SDR preset declares the matching Bayer/bit-depth/geometry and little-endian uint16 storage.",
            ))
        else:
            raise ValueError(f"no explicit corpus assignment: {source}")
    base = "in_frames/updated_images"
    facts = json.loads((ROOT / "tmp/gpu_validation/m5/updated_decoder_frozen_facts.json").read_text())
    for row in facts:
        for name in ("raw", "tiff", "sidecar"):
            if file_identity(Path(row[name]["path"])) != row[name]:
                raise RuntimeError("frozen RAW/TIFF decoder evidence changed")
    ready.extend((
        _case(
            "updated_hdr16_raw_le", base + "/TRI054S-C_244800692__260811155533_image11.raw",
            [base + "/configs/imx490_hdr_16.yml"],
            overlay={"sensor_info": {
                "source_format": "raw", "storage_format": "uint16",
                "endian_type": "ieee-le", "data_alignment": "LSB",
            }, "companding": {"is_enable": False}},
            evidence="Matching BayerRG16 sidecar and TIFF RGGB/YX metadata. Frozen exact comparison proves RAW little-endian uint16 equals corresponding TIFF; big-endian does not. Original preset also contains invalid companding.is_enable='fasle'. This is an explicit companding-disabled/LE validation variant, not qualification or modification of the invalid original preset.",
            extra_inputs=["tmp/gpu_validation/m5/updated_decoder_frozen_facts.json",
                          base + "/TRI054S-C_244800692__260811155533_image11.raw.json",
                          base + "/tiffs/TRI054S-C_244800692__260811155533_image11.tiff"],
        ),
        _case(
            "updated_hdr12_unpacked_tiff",
            base + "/tiffs/TRI054S-C_244800692__260811155601_image12.tiff",
            [base + "/configs/imx490_hdr_12.yml"],
            overlay={"sensor_info": {
                "source_format": "tiff", "storage_format": "uint16", "data_alignment": "LSB",
            }, "companding": {"is_enable": False}},
            evidence="Original TIFF explicitly declares YX/RGGB/BayerRG12p metadata but stores unpacked uint16 samples. Only this TIFF representation is evaluated, not the unsupported packed RAW. Preset companding.is_enable='fasle' is invalid; this is an explicit companding-disabled validation variant and the original preset remains invalid.",
            extra_inputs=["tmp/gpu_validation/m5/updated_decoder_frozen_facts.json",
                          base + "/TRI054S-C_244800692__260811155601_image12.raw.json"],
        ),
        _case(
            "updated_sdr8_raw", base + "/TRI054S-C_244800692__260811155628_image13.raw",
            [base + "/configs/imx490_sdr.yml"],
            evidence="BayerRG8 sidecar and corresponding RGGB/YX TIFF agree; frozen exact uint8 RAW/TIFF comparison passes. Preset declares RGGB/8-bit/uint8/2880x1860 without an encoding overlay.",
            extra_inputs=["tmp/gpu_validation/m5/updated_decoder_frozen_facts.json",
                          base + "/TRI054S-C_244800692__260811155628_image13.raw.json",
                          base + "/tiffs/TRI054S-C_244800692__260811155628_image13.tiff"],
        ),
    ))
    repair = copy.deepcopy(next(case for case in ready if case["id"] == "updated_hdr24_raw"))
    repair["id"] = "updated_hdr24_green_repair"
    repair["config"]["sensor_info"]["repair_zero_green"] = True
    repair["validation_overlay"]["sensor_info"]["repair_zero_green"] = True
    repair["assignment_evidence"] += " Additional explicit green-repair-enabled integration variant."
    ready.append(repair)
    return {"ready": ready, "reused": reused, "blocked": blocked,
            "original_preset_limitations": [
                {"config": "config/triton_lauren_hdr24_cam.yml",
                 "reason": "Incomplete preset requires missing base_hdr.yml; existing qualified HDR24 composition is an explicit recorded validation configuration."},
                {"config": "in_frames/updated_images/configs/imx490_hdr_24.yml",
                 "reason": "Schema rejects endian alias little; uint32 declaration mismatches BayerRG24 3-byte storage. Explicit packed24/ieee-le variant only."},
                {"config": "in_frames/updated_images/configs/imx490_hdr_16.yml",
                 "reason": "Invalid companding.is_enable='fasle'; BE declaration disagrees with exact paired TIFF proof. Explicit companding-disabled/LE variant only."},
                {"config": "in_frames/updated_images/configs/imx490_hdr_12.yml",
                 "reason": "Invalid companding.is_enable='fasle'; packed RAW unsupported. Explicit companding-disabled unpacked TIFF variant only."},
            ],
            "scope": "All 17 original inventoried RAW inputs accounted for, plus two unpacked TIFF representations and updated SDK 8/16-bit captures. AD/SVS assignments are declared camera-family profiles, not independent sensor encoding certification."}


def full_algorithm_families():
    return [
        _case(
            "fv_full_" + algorithm, "in_frames/hdr_mode/FV.raw", ["config/SVS_cam.yml"],
            overlay={"demosaic": {"algorithm": algorithm}},
            evidence="Original full-resolution 1920x1536 FV frame with a recorded algorithm-only overlay; all other camera preset processing retained.",
        )
        for algorithm in ("hamilton_adams_opt", "ppg_opt", "lmmse_fast", "ahd_opt")
    ]


def algorithm_rois(datasets: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((datasets / "datasets.json").read_text())
    row = next(value for value in metadata["datasets"] if value["id"] == "sdr_colorchecker")
    for identity in (row["raw"], row["config"], row["pixels"], row["original_source"]):
        if file_identity(Path(identity["path"])) != identity:
            raise RuntimeError("frozen real ROI source changed")
    data = np.load(row["pixels"]["path"], allow_pickle=False)[192:320, 192:320].copy()
    source = output / "colorchecker_center_128.raw"
    with source.open("xb") as stream:
        stream.write(data.tobytes(order="C"))
    config = yaml.safe_load(Path(row["config"]["path"]).read_text())
    config["sensor_info"].update(width=128, height=128)
    config["platform"]["filename"] = source.name
    cases = []
    for algorithm in DEMOSAIC_ALGORITHMS:
        variant = copy.deepcopy(config)
        variant["demosaic"]["algorithm"] = algorithm
        cases.append({
            "id": "real_roi_" + algorithm, "source": str(source.resolve()), "config": variant,
            "assignment_evidence": "Fixed center 128x128 real ColorChecker ROI selected before numerical evaluation, not full-frame coverage. Even-origin extraction preserves RGGB.",
            "original_bbox_yxhw": [row["bbox_yxhw"][0] + 192,
                                   row["bbox_yxhw"][1] + 192, 128, 128],
            "input_identities": [file_identity(source), row["raw"], row["config"], row["original_source"]],
        })
    return cases


def interaction_rois():
    path = ROOT / "tmp/gpu_validation/m5/real_demosaic_66b12e45/real_roi_bilinear/case.json"
    base = json.loads(path.read_text())
    variants = (
        ("reinhard", "fixed", "pre_demosaic", "post_transfer", True, True,
         {"model": "ycbcr", "format": "yuv", "bit_depth": 16, "layout": "422",
          "packing": "uyvy", "matrix": "bt709", "range": "limited"}),
        ("durand", "cpu", "post_ccm", "post_transfer", True, False,
         {"model": "rgb", "format": "tiff", "bit_depth": 16, "transfer": "rec709"}),
        ("durand", "cpu", "pre_demosaic", None, True, True,
         {"model": "rgb", "format": "png", "bit_depth": 16, "transfer": "srgb"}),
        ("aces_fitted", "float", "post_ccm", "pre_transfer", False, False,
         {"model": "ycbcr", "format": "yuv", "bit_depth": 8, "layout": "444",
          "matrix": "bt709", "range": "full"}),
        ("aces_fitted", "lut", "pre_demosaic", None, True, False,
         {"model": "ycbcr", "format": "yuv", "bit_depth": 16, "layout": "444",
          "matrix": "bt601", "range": "full"}),
        ("hable", "float", "post_ccm", "post_transfer", True, False,
         {"model": "rgb", "format": "png", "bit_depth": 16, "transfer": "power", "gamma": 2.4}),
        ("hable", "lut", "pre_demosaic", None, True, True,
         {"model": "rgb", "format": "png", "bit_depth": 8, "transfer": "rec709"}),
        ("reinhard", "lut", "post_ccm", "pre_transfer", False, False,
         {"model": "rgb", "format": "jpeg", "bit_depth": 8, "transfer": "srgb"}),
        ("reinhard", "float", "post_ccm", "post_transfer", True, False,
         {"model": "rgb", "format": "png", "bit_depth": 8, "transfer": "srgb"}),
    )
    cases = []
    for index, (operator, backend, placement, ldci, nlm, crop, output) in enumerate(variants):
        case = copy.deepcopy(base)
        case["id"] = f"interaction_{index:02d}_{operator}_{backend}_{placement}"
        case["input_identities"].append(file_identity(path))
        config = case["config"]
        config["schema_version"] = 2
        config["tone_mapping"] = {
            "is_enable": True, "operator": operator, "backend": backend,
            "placement": placement, "normalize_output": operator != "durand",
        }
        config["output"] = output
        config["lens_shading_correction"]["is_enable"] = True
        config["sharpen"]["is_enable"] = True
        config["ldci"].update(is_enable=ldci is not None,
                              placement=ldci or "post_transfer", wind=16, clip_limit=2.0)
        config["2d_noise_reduction"].update(
            is_enable=nlm, window_size=3, patch_size=3, strength=10.0 / 255.0,
        )
        config["crop"].update(
            is_enable=crop, crop_x_start=1, crop_y_start=1, new_height=126, new_width=126,
        )
        config["scale"].update(is_enable=index in (0, 4), new_height=64, new_width=64)
        case["assignment_evidence"] = (
            "Fixed real ColorChecker 128x128 source with explicit combined-block validation settings. "
            "LSC/sharpen enabled, selected NLM/CLAHE, all tone-map algorithm/backend pairings, "
            "pre/post placements, selected odd-origin CFA crops and scaling/output variants. "
            "These are integration overlays and ROI coverage, not original-preset calibration claims."
        )
        cases.append(case)
    return cases
