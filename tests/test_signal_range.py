from __future__ import annotations

import pytest

from util.signal_range import linear_code_range


@pytest.mark.parametrize("pwc,custom,bits,expected,origin", (
    (False, False, None, 12, "sensor"),
    (True, False, None, 24, "decompanded"),
    (False, True, None, 12, "oecf_lut"),
    (True, True, None, 12, "oecf_lut"),
    (True, True, 24, 24, "oecf_lut"),
    (False, True, 32, 32, "oecf_lut"),
))
def test_active_declared_stages_own_range_not_unused_hdr_budget(pwc, custom, bits, expected, origin) -> None:
    config = {
        "sensor_info": {"bit_depth": 12, "hdr_bit_depth": 24},
        "companding": {"is_enable": pwc, "companded": False},
        "oecf": {"is_enable": True, "r_lut": [0, 1] if custom else None},
    }
    if bits is not None:
        config["oecf"]["output_bit_depth"] = bits
    resolved = linear_code_range(config)
    assert resolved.bit_depth == expected
    assert resolved.maximum == 2**expected - 1
    assert resolved.origin == origin
    assert config["sensor_info"]["hdr_bit_depth"] == 24


def test_disabled_oecf_does_not_change_the_reference_white() -> None:
    config = {
        "sensor_info": {"bit_depth": 12, "hdr_bit_depth": 24},
        "companding": {"is_enable": True},
        "oecf": {"is_enable": False, "r_lut": [0, 1], "output_bit_depth": 8},
    }
    assert linear_code_range(config).bit_depth == 24


@pytest.mark.parametrize("value", (0, 33, 12.5, True, "12", float("inf")))
def test_active_range_does_not_guess_or_truncate_invalid_declarations(value) -> None:
    with pytest.raises(ValueError, match="bit_depth"):
        linear_code_range({"sensor_info": {"bit_depth": value}})
