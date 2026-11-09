import numpy as np
import pytest

from modules.tone_mapping.native import map_native, map_native_max_rgb
from modules.tone_mapping.registry import ToneMappingError, ToneMappingSpec


@pytest.mark.parametrize("operator", ("reinhard", "hable", "aces_fitted"))
def test_max_rgb_keeps_saturated_channels_in_range_and_preserves_hue(operator):
    spec = ToneMappingSpec(operator, backend="float")
    rgb = np.array([[[0.02, 0.05, 0.9], [0.4, 0.3, 0.05], [0.3, 0.3, 0.3]]], dtype=np.float32) * 8.0
    out = np.asarray(map_native_max_rgb(rgb, config=spec), dtype=np.float64)
    lum_out = np.asarray(map_native(rgb, config=spec), dtype=np.float64)
    assert out.max() <= 1.0 + 1e-6
    assert lum_out[0, 0].max() > out[0, 0].max()
    ratio = out / rgb
    assert np.allclose(ratio, ratio[..., :1], rtol=1e-5)
    assert np.all(np.diff(out[0, 2]) == 0)


def test_max_rgb_black_and_validation():
    spec = ToneMappingSpec("reinhard", backend="float")
    out = np.asarray(map_native_max_rgb(np.zeros((2, 2, 3), np.float32), config=spec))
    assert np.all(out == 0)
    with pytest.raises(ToneMappingError):
        map_native_max_rgb(np.zeros((2, 2), np.float32), config=spec)
