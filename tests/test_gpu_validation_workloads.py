import numpy as np
import pytest

from tools.gpu_validation.workloads import (
    _read_artifact, compare_feedback, output_variants,
)
from util.native_output import output_metadata
from util.output_contract import OutputSpec
from util.output_writer import write_output


def test_output_workload_variants_are_valid_and_distinct():
    variants = output_variants()
    assert len(variants) == len({name for name, _ in variants}) == 10
    specs = {name: OutputSpec.from_mapping(mapping) for name, mapping in variants}
    assert specs["linear_f32"].floating_linear
    assert specs["uyvy16"].packing == "uyvy" and specs["uyvy16"].bit_depth == 16
    assert specs["yuyv8"].packing == "yuyv" and specs["yuyv8"].layout == "422"


def test_feedback_preserves_statistics_and_discrete_contracts():
    expected = {"awb_gains": (1.0, 2.0), "ae_meter_average": .3, "gain_index": 2}
    actual = {"awb_gains": (1.0 + 1e-11, 2.0), "ae_meter_average": .3, "gain_index": 2}
    assert compare_feedback(expected, actual)["passed"]
    assert not compare_feedback(expected, {**actual, "gain_index": True})["passed"]
    assert not compare_feedback(expected, {**actual, "awb_gains": (1.0001, 2.0)})["passed"]
    assert not compare_feedback(expected, {"awb_gains": (1.0, 2.0)})["passed"]


@pytest.mark.parametrize("name", (
    "png8", "png16", "tiff16", "linear_f32", "yuv444_8", "yuv444_16", "yuyv8", "uyvy16",
))
def test_real_output_harness_reader_exact_native_values(tmp_path, name):
    spec = OutputSpec.from_mapping(dict(output_variants())[name])
    if spec.floating_linear:
        native = np.array([[[-.125, .5, 2.0], [0., 1., 1.25]]], np.float32)
    else:
        dtype = np.uint8 if spec.bit_depth == 8 else np.uint16
        native = np.array([[[1, 129, 255], [2, 11, 13]]], dtype=dtype)
        if spec.bit_depth == 16:
            native *= 257
        if spec.layout == "422":
            native = np.array([[1, 2, 129, 255]], dtype=dtype)
    metadata = output_metadata(native, spec, source_identity="fixture", tone_mapped=False)
    artifact = write_output(
        native, spec, metadata, output_dir=tmp_path, stem=name,
        effective_config={}, migration_report={},
    )[0]
    actual = _read_artifact(artifact, spec, native.shape)
    assert actual.dtype == native.dtype
    np.testing.assert_array_equal(actual, native)
