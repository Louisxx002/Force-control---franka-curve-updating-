from dataclasses import replace
import math

import numpy as np
import pytest

from curve_wipe.force import (ForceSafetyStop, NormalAdmittance,
                              NormalAdmittanceConfig, compensate_wrench,
                              normal_force, simulate_normal_force)


def transform(rotation=None, translation=(0, 0, 0)):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def parameters():
    return dict(T_base_ee=np.eye(4), T_ee_sensor=np.eye(4),
                T_ee_tcp=np.eye(4), bias_sensor_6=np.zeros(6),
                gravity_base_N=np.zeros(3), com_sensor_m=np.zeros(3), sensor_sign=1)


@pytest.mark.parametrize("sign", [-1, 1])
def test_six_axis_gravity_compensation_across_independent_tool_poses(sign):
    rng = np.random.default_rng(31)
    bias = np.array([.8, -.3, .4, .02, -.03, .05])
    gravity = np.array([0., 0., -2.3])
    com = np.array([.02, -.01, .07])
    mount = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for _ in range(20):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        sensor_rotation = rotation @ mount
        gravity_sensor = sensor_rotation.T @ gravity
        raw = bias + sign * np.r_[gravity_sensor, np.cross(com, gravity_sensor)]
        result = compensate_wrench(raw, T_base_ee=transform(rotation),
                                   T_ee_sensor=transform(mount, [.01, .02, 0]),
                                   T_ee_tcp=transform(translation=[0, 0, .12]),
                                   bias_sensor_6=bias, gravity_base_N=gravity,
                                   com_sensor_m=com, sensor_sign=sign)
        np.testing.assert_allclose(result.wrench_base_at_tcp_6, np.zeros(6), atol=1e-12)


def test_moment_origin_shift_to_tcp_with_sensor_rotation():
    params = parameters()
    params["T_ee_sensor"] = transform(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    params["T_ee_tcp"] = transform(translation=[.1, 0, 0])
    # Base force [0,0,2] applied at TCP yields sensor torque [0,-.2,0]
    # which is [-.2,0,0] in the rotated sensor frame.
    result = compensate_wrench([0, 0, 2, -.2, 0, 0], **params)
    np.testing.assert_allclose(result.force_base_N, [0, 0, 2])
    np.testing.assert_allclose(result.torque_at_tcp_base_Nm, [0, 0, 0], atol=1e-12)


def test_sensor_sign_and_positive_compressive_normal():
    params = parameters()
    params["sensor_sign"] = -1
    result = compensate_wrench([0, 0, -1.7, 0, 0, 0], **params)
    assert normal_force(result, [0, 0, 1]) == pytest.approx(1.7)
    params["sensor_sign"] = 0
    with pytest.raises(ValueError, match="sensor_sign"):
        compensate_wrench(np.zeros(6), **params)


def test_invalid_transform_and_nonfinite_raw_rejected():
    params = parameters()
    params["T_ee_sensor"][0, 0] = -1
    with pytest.raises(ValueError, match="proper rotation"):
        compensate_wrench(np.zeros(6), **params)
    with pytest.raises(ValueError, match="finite"):
        compensate_wrench([math.nan] * 6, **parameters())


def step(core, force=(0, 0, 0), torque=(0, 0, 0), normal=(0, 0, 1),
         dt=.005, age=0):
    return core.step(force, torque, normal, dt_s=dt, sample_age_s=age)


def test_low_force_moves_into_surface_and_high_force_moves_outward():
    low = step(NormalAdmittance(), normal=[1, 0, 0])
    assert low.delta_base_m[0] < 0
    np.testing.assert_equal(low.delta_base_m[1:], [0, 0])
    high = step(NormalAdmittance(), force=[0, 0, 2])
    assert high.delta_base_m[2] > 0
    assert abs(low.velocity_m_s) <= .003


@pytest.mark.parametrize("kwargs,reason", [
    ({"force": [0, 0, math.nan]}, "finite"),
    ({"torque": [math.inf, 0, 0]}, "finite"),
    ({"force": [8, 0, 0]}, "force norm"),
    ({"torque": [0, .4, 0]}, "torque norm"),
    ({"age": .051}, "stale"),
    ({"age": -1}, "stale"),
    ({"age": math.nan}, "stale"),
    ({"dt": .021}, "dt_s"),
    ({"dt": 0}, "dt_s"),
    ({"normal": [0, 0, 0]}, "unit"),
    ({"force": [0, 0, -1]}, "sensor sign"),
])
def test_faults_stop_before_motion_and_latch(kwargs, reason):
    core = NormalAdmittance()
    with pytest.raises(ForceSafetyStop, match=reason):
        step(core, **kwargs)
    assert core.offset_m == 0 and core.velocity_m_s == 0
    with pytest.raises(ForceSafetyStop, match="latched"):
        step(core)


def test_analytic_spring_converges_and_respects_velocity_bound():
    result = simulate_normal_force()
    report = result["report"]
    assert report["converged"] and report["status"] == "completed"
    assert abs(report["final_error_N"]) < .01
    assert report["final_offset_m"] == pytest.approx(-.001 - 1 / 800, abs=2e-5)
    assert report["max_abs_velocity_m_s"] <= .003
    assert result["trajectory"][0]["commanded_offset_m"] < 0


def test_zero_force_eventually_stops_without_crossing_travel_limit():
    report = simulate_normal_force(spring_stiffness_N_m=0)["report"]
    assert report["status"] == "stopped" and not report["converged"]
    assert "displacement" in report["stop_reason"]
    assert abs(report["final_offset_m"]) < .008


def test_excessive_simulation_dt_cannot_claim_success():
    report = simulate_normal_force(dt_s=.1)["report"]
    assert report["status"] == "stopped" and report["steps"] == 0


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        replace(NormalAdmittanceConfig(), max_velocity_m_s=math.nan)
    with pytest.raises(ValueError):
        replace(NormalAdmittanceConfig(), target_force_N=9)
