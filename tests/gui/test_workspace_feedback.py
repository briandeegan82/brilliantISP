from __future__ import annotations

import pytest

from tools.gui.feedback import feedback_edits


def test_manual_awb_edits_are_explicit_and_do_not_modify_inputs() -> None:
    config, feedback = {}, {"awb_status": "estimated", "awb_gains": [1.5, 2.0]}
    assert feedback_edits(config, feedback, "awb") == {
        "white_balance.r_gain": 1.5, "white_balance.b_gain": 2.0,
        "white_balance.is_enable": True, "white_balance.is_auto": False,
        "auto_white_balance.is_enable": False,
    }
    assert config == {} and feedback == {"awb_status": "estimated", "awb_gains": [1.5, 2.0]}
    with pytest.raises(ValueError):
        feedback_edits({}, dict(feedback, awb_status="fallback"), "awb")
    for gains in ([0, 1], [float("inf"), 1], [True, 1], [1]):
        with pytest.raises(ValueError):
            feedback_edits({}, dict(feedback, awb_gains=gains), "awb")


def test_manual_gain_freezes_applied_not_suggested_gain() -> None:
    config = {"digital_gain": {"gain_array": [1, 2, 4], "is_enable": False}}
    feedback = {"digital_gain_index": 1, "ae_suggested_gain_index": 2, "next_digital_gain_index": 2}
    assert feedback_edits(config, feedback, "gain") == {
        "digital_gain.current_gain": 1, "digital_gain.is_auto": False,
        "auto_exposure.is_enable": False,
    }
    assert config["digital_gain"]["is_enable"] is False
    for index in (-1, 3, True, 1.5, None):
        with pytest.raises(ValueError):
            feedback_edits(config, {"digital_gain_index": index}, "gain")
