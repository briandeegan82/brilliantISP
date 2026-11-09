import math

import numpy as np
import pytest

from modules.auto_exposure import smooth_control as sc
from modules.auto_white_balance import temporal


def test_params_defaults_and_validation():
    p = sc.SmoothAEParams.from_config({})
    assert p == sc.SmoothAEParams()
    with pytest.raises(ValueError):
        sc.SmoothAEParams.from_config({"output_target": 1.5})
    with pytest.raises(ValueError):
        sc.SmoothAEParams.from_config({"gain_min": 10.0, "gain_max": 1.0})


def test_solve_gain_hits_target_through_tone_map():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.0, 1e-3, size=4096)
    p = sc.SmoothAEParams(output_target=0.2, gain_min=1.0, gain_max=1e6)

    def tm(v):
        return v / (1.0 + v)

    g = sc.solve_gain(y, tm, p)
    assert np.mean(np.clip(tm(y * g), 0, 1)) == pytest.approx(0.2, rel=1e-3)


def test_solve_gain_saturates_at_limits():
    p = sc.SmoothAEParams(output_target=0.5, gain_min=1.0, gain_max=4.0)
    assert sc.solve_gain(np.full(16, 1e-9), lambda v: v, p) == 4.0
    assert sc.solve_gain(np.full(16, 10.0), lambda v: v, p) == 1.0
    assert sc.solve_gain(np.zeros(16), lambda v: v, p) == 4.0


def test_next_gain_deadband_smoothing_and_slew():
    # alpha 0.5 and 0.25 stop/frame at 10 fps
    p = sc.SmoothAEParams(time_constant_s=1 / (10 * math.log(2)), deadband_stops=0.1,
                          max_rate_stops_per_s=2.5, frame_rate=10.0)
    assert p.alpha == pytest.approx(0.5) and p.max_step_stops == pytest.approx(0.25)
    assert sc.next_gain(None, 8.0, p) == 8.0
    assert sc.next_gain(8.0, 8.0 * 2 ** 0.05, p) == 8.0
    # 1 stop error: (1 - 0.1) * 0.5 = 0.45 stop -> slew-limited to 0.25 stop
    assert math.log2(sc.next_gain(8.0, 16.0, p) / 8.0) == pytest.approx(0.25)
    # 0.3 stop error: (0.3 - 0.1) * 0.5 = 0.1 stop
    assert math.log2(sc.next_gain(8.0, 8.0 * 2 ** -0.3, p) / 8.0) == pytest.approx(-0.1)


def test_next_gain_is_frame_rate_independent():
    def settle(fps):
        p = sc.SmoothAEParams(time_constant_s=0.5, deadband_stops=0.0, max_rate_stops_per_s=100.0,
                              frame_rate=fps)
        g = 1.0
        for _ in range(int(fps)):  # one second
            g = sc.next_gain(g, 16.0, p)
        return math.log2(g)

    assert settle(10.0) == pytest.approx(settle(30.0), rel=1e-9)
    assert settle(30.0) == pytest.approx(4.0 * (1 - math.exp(-2.0)))


def test_scene_luminance_unity_gain():
    rgb = np.ones((4, 4, 3)) * 8.0
    y = sc.scene_luminance(rgb, 4.0, 2)
    assert y.shape == (2, 2)
    assert np.allclose(y, 2.0)


def test_awb_temporal_inactive_without_keys():
    assert not temporal.is_active({"is_enable": True, "algorithm": "pca"})
    assert temporal.is_active({"time_constant_s": 0.5})


def test_awb_temporal_clamp_and_filter():
    half = {"time_constant_s": 1 / (10 * math.log(2)), "frame_rate": 10.0}
    parm = {"r_gain_range": [1.0, 2.0], "b_gain_range": [1.0, 2.0], "max_rate_per_s": 1.0, **half}
    assert temporal.filter_gains((3.0, 0.5), None, parm) == (2.0, 1.0)
    r, b = temporal.filter_gains((2.0, 1.0), (1.0, 2.0), parm)
    assert math.log(r) == pytest.approx(0.1)
    assert math.log(b) == pytest.approx(math.log(2.0) - 0.1)
    r, b = temporal.filter_gains((1.1, 1.0), (1.0, 1.0), half)
    assert r == pytest.approx(math.sqrt(1.1))
    with pytest.raises(ValueError):
        temporal.filter_gains((1.0, 1.0), (1.0, 1.0), {"time_constant_s": -1.0})


def test_awb_low_light_prior_blend():
    parm = {"low_light_prior": [2.0, 1.5], "low_light_gain_range": [100.0, 10000.0]}
    assert temporal.low_light_weight(10.0, parm) == 0.0
    assert temporal.low_light_weight(1000.0, parm) == pytest.approx(0.5)
    assert temporal.low_light_weight(1e6, parm) == 1.0
    assert temporal.filter_gains((1.0, 3.0), None, parm, input_gain=1e6) == pytest.approx((2.0, 1.5))
    r, b = temporal.filter_gains((1.0, 3.0), None, parm, input_gain=1000.0)
    assert r == pytest.approx(math.sqrt(2.0)) and b == pytest.approx(math.sqrt(4.5))
    assert temporal.low_light_weight(1e6, {}) == 0.0


def test_low_light_target_rolloff_lowers_night_target_only():
    p = sc.SmoothAEParams(output_target=0.16, low_light_target_stops=1.0, low_light_gain_range=(1e3, 1e4))
    assert p.target_at(100.0) == pytest.approx(0.16)
    assert p.target_at(math.sqrt(1e7)) == pytest.approx(0.16 * 2 ** -0.5)
    assert p.target_at(1e6) == pytest.approx(0.08)
    y = np.full(64, 1e-5)
    g = sc.solve_gain(y, lambda v: v, p)
    assert g * 1e-5 == pytest.approx(p.target_at(g), rel=1e-3)
    assert g < 0.16 / 1e-5


def test_clip_fraction_cap_limits_gain():
    y = np.concatenate([np.full(90, 0.01), np.full(10, 0.5)])
    free = sc.solve_gain(y, lambda v: np.minimum(v, 1.0), sc.SmoothAEParams(output_target=0.18, gain_max=1e3))
    p = sc.SmoothAEParams(output_target=0.18, gain_max=1e3, clip_level=0.98, max_clip_fraction=0.05,
                          clip_max_backoff_stops=10.0)
    capped = sc.solve_gain(y, lambda v: np.minimum(v, 1.0), p)
    assert free * 0.5 >= 0.98
    assert capped * 0.5 < 0.98
    with pytest.raises(ValueError):
        sc.SmoothAEParams.from_config({"max_clip_fraction": 0.0})


def test_clip_cap_backoff_is_bounded():
    y = np.concatenate([np.full(90, 0.01), np.full(10, 0.5)])
    tm = lambda v: np.minimum(v, 1.0)  # noqa: E731
    free = sc.solve_gain(y, tm, sc.SmoothAEParams(output_target=0.18, gain_max=1e3))
    p = sc.SmoothAEParams(output_target=0.18, gain_max=1e3, max_clip_fraction=0.05, clip_max_backoff_stops=0.5)
    assert sc.solve_gain(y, tm, p) == pytest.approx(free * 2 ** -0.5, rel=1e-6)
