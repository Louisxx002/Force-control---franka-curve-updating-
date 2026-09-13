import numpy as np
import pytest
from curve_wipe.approach_checks import (validate_config, flange_waypoints,
    check_fk, check_trajectory, dense_joint_samples)


def test_flange_conversion_uses_full_ee_transform():
    flange = np.eye(4)
    flange[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    flange[:3, 3] = [.02, -.01, .1034]
    ee = np.eye(4)
    ee[:3, 3] = [.4, .2, .5]
    output = flange_waypoints([ee], flange)[0]
    np.testing.assert_allclose(output@flange, ee, atol=1e-12)


def test_fk_mismatch_rejects_wrong_model_or_frame():
    wrong = np.eye(4)
    wrong[2, 3] = .01
    with pytest.raises(ValueError, match='mismatch'):
        check_fk(np.eye(4), wrong)


def trajectory():
    q = np.zeros((3, 7))
    q[:, 0] = [0, .02, .04]
    return ['j'+str(i) for i in range(7)], q, [0, 1, 2]


@pytest.mark.parametrize('failure', ['names', 'start', 'jump', 'nan', 'time', 'joint2'])
def test_unsafe_trajectory_results_rejected(failure):
    names, q, t = trajectory()
    returned = names.copy()
    if failure == 'names': returned.reverse()
    if failure == 'start': q[0, 0] = .01
    if failure == 'jump': q[-1, 0] = .3
    if failure == 'nan': q[-1, 0] = np.nan
    if failure == 'time': t[-1] = 1
    if failure == 'joint2': q[:, 1] = [1.70, 1.71, 1.72]
    start = q[0].copy() if failure == 'joint2' else np.zeros(7)
    with pytest.raises(ValueError):
        check_trajectory(returned, q, t, names, start)


def test_densification_and_valid_trajectory():
    names, q, t = trajectory()
    assert check_trajectory(names, q, t, names, np.zeros(7))['points'] == 3
    dense = np.array(list(dense_joint_samples(q)))
    np.testing.assert_allclose(dense[0], q[0])
    np.testing.assert_allclose(dense[-1], q[-1])
    assert np.max(np.abs(np.diff(dense, axis=0))) <= .010000001


def test_missing_environment_configuration_rejected():
    with pytest.raises(ValueError, match='configuration'):
        validate_config({'group': 'fr3_arm'})
