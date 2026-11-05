import pytest

from tools.gpu_validation.corpus import case_definitions, run_corpus


def test_corpus_case_definitions_are_explicit_and_unique():
    cases = case_definitions()
    assert {case["id"] for case in cases} == {
        "sdr_colorchecker", "triton_sdr_tiff", "triton_hdr24_overlay",
    }
    assert len(cases) == len({case["id"] for case in cases})
    assert all(case["source"] and case["configs"] and case["notes"] for case in cases)
    hdr = next(case for case in cases if case["id"] == "triton_hdr24_overlay")
    assert hdr["overlay"]["sensor_info"]["storage_format"] == "packed24"
    assert len(hdr["configs"]) == 2


def test_unknown_corpus_selection_does_not_create_artifacts(tmp_path):
    output, scratch = tmp_path / "output", tmp_path / "scratch"
    with pytest.raises(ValueError, match="unknown corpus case ids"):
        run_corpus(output, scratch, "cuda:0", case_ids=("not_a_case",))
    assert not output.exists() and not scratch.exists()
