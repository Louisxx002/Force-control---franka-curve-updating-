from types import SimpleNamespace

import numpy as np
import pytest

from curve_wipe.execute import (ExecutionError, NormalAdmittance, check_robot_state,
                                motion_step, MAX_LOOP_DT, pass_distance, WIPE_RAMP_TIME_S)


def state(mode=2, error=False, historic=False):
    return SimpleNamespace(robot_mode=mode,
        current_errors=SimpleNamespace(communication_constraints_violation=error),
        last_motion_errors=SimpleNamespace(joint_reflex=historic),
        control_command_success_rate=.59)


def test_healthy_move_and_historic_fault_do_not_stop():
    check_robot_state(state(historic=True), (1, 2))


def test_fault_while_still_move_preserves_real_cause():
    log = {}
    with pytest.raises(ExecutionError, match='communication_constraints_violation'):
        check_robot_state(state(error=True), (1, 2), log, 'wipe_backward')
    assert log['robot_fault']['phase'] == 'wipe_backward'
    assert log['robot_fault']['control_command_success_rate'] == .59


def test_reflex_mode_without_error_bits_still_stops():
    with pytest.raises(ExecutionError):
        check_robot_state(state(mode=4), (1, 2))


def test_actual_pybind_empty_errors_are_not_a_fault():
    franky = pytest.importorskip('franky')
    s = state()
    s.robot_mode = franky.RobotMode.Move
    s.current_errors = s.last_motion_errors = franky.Errors()
    check_robot_state(s, (franky.RobotMode.Idle.value, franky.RobotMode.Move.value))


def test_delay_does_not_produce_catchup_command():
    assert motion_step(.057) == .01
    with pytest.raises(ExecutionError):
        motion_step(MAX_LOOP_DT + .001)


def test_force_step_has_continuous_bounded_normal_velocity():
    a = NormalAdmittance(-.0015)
    height = 0.
    for force in [0.] * 50 + [8.] * 50 + [1.] * 50:
        old_velocity = a.velocity
        next_height = a.step(height, force, .01)
        assert abs(a.velocity) <= .002
        assert abs(a.velocity - old_velocity) / .01 <= .021
        assert abs(next_height - height) <= .002 * .01
        height = next_height
