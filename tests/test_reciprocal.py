import numpy as np
from curve_wipe.execute import WIPE_RAMP_TIME_S, pass_distance


def test_roundtrip_endpoints_and_speed():
    length = .15
    duration = 1.875 * length / .003
    t = np.linspace(0, duration, 10001)
    forward = np.array([pass_distance(x, duration, length) for x in t])
    backward = np.array([pass_distance(x, duration, length, True) for x in t])
    assert forward[0] == 0 and forward[-1] == length
    assert backward[0] == length and backward[-1] == 0
    assert np.max(np.diff(forward)/np.diff(t)) <= .00300001
    assert abs((backward[1]-backward[0])/(t[1]-t[0])) < 1e-6
    assert abs((forward[-1]-forward[-2])/(t[1]-t[0])) < 1e-6


def test_wipe_has_constant_cruise_speed_between_smooth_ramps():
    length, speed = .15, .0045
    duration = length / speed + WIPE_RAMP_TIME_S
    t = np.linspace(WIPE_RAMP_TIME_S + .1, duration - WIPE_RAMP_TIME_S - .1, 100)
    ds = np.diff([pass_distance(x, duration, length) for x in t])
    np.testing.assert_allclose(ds / np.diff(t), speed, rtol=1e-5, atol=1e-8)
