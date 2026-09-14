import ast
from pathlib import Path

import numpy as np
import pytest

from curve_wipe.execute import MAX_TENSILE_NORMAL_N, TOTAL_FORCE_LIMIT_N, check_wrench, ExecutionError

def test_execute_source_has_no_removed_pose_after_reference():
    source=Path('curve_wipe/execute.py').read_text()
    assert 'pose_after' not in source
    ast.parse(source)


def test_hardware_reverse_force_threshold_is_1_5_newtons():
    assert MAX_TENSILE_NORMAL_N == pytest.approx(1.5)
    force, _ = check_wrench(np.array([0, 0, MAX_TENSILE_NORMAL_N, 0, 0, 0.]), np.zeros(6))
    assert force == pytest.approx(-1.5)
    with pytest.raises(ExecutionError, match="below -1.5 N"):
        check_wrench(np.array([0, 0, 1.51, 0, 0, 0.]), np.zeros(6))


def test_contact_torque_limit_uses_vector_magnitude():
    from curve_wipe.execute import CONTACT_TORQUE_LIMIT_NM
    assert CONTACT_TORQUE_LIMIT_NM == pytest.approx(1.5)
    for torque in ([1.5, 0, 0], [0, -1.5, 0], [0, 0, 1.5]):
        check_wrench(np.array([0., 0, 0, *torque]), np.zeros(6))
    for torque in ([1.501, 0, 0], [0, -1.501, 0], [1.07, 1.07, 0]):
        with pytest.raises(ExecutionError, match="1.5 Nm"):
            check_wrench(np.array([0., 0, 0, *torque]), np.zeros(6))


def test_total_force_limit_is_8_newtons():
    assert TOTAL_FORCE_LIMIT_N == pytest.approx(8.0)
    check_wrench(np.array([8.0, 0, 0, 0, 0, 0]), np.zeros(6))
    with pytest.raises(ExecutionError, match="exceeds 8 N"):
        check_wrench(np.array([8.001, 0, 0, 0, 0, 0]), np.zeros(6))
