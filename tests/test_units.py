import numpy as np
import pytest
import struct
from curve_wipe.serial_frames import parse_frame, HEADER, TERMINATOR
from curve_wipe.convert_payload_units import convert
from curve_wipe.adaptive import ForceFeedback, Settings


def test_wire_known_force_and_torque_are_si():
    f, m = parse_frame(HEADER + struct.pack('<6f', 1, -2, 0, .5, 0, -1) + TERMINATOR)
    np.testing.assert_allclose(f, [9.80665, -19.6133, 0])
    np.testing.assert_allclose(m, [4.903325, 0, -9.80665])


def test_legacy_conversion_preserves_time_robot_state_and_source():
    s = dict(raw_mean_6=[1]*6, raw_std_6=[.1]*6,
             raw_samples=[[1234]+[1]*6], robot_states=[{'force': 7}])
    d = dict(schema_version=1, train=[s], holdout=[])
    c = convert(d)
    assert d['train'][0]['raw_mean_6'] == [1]*6
    assert c['train'][0]['raw_samples'][0] == [1234]+[9.80665]*6
    assert c['train'][0]['robot_states'] == s['robot_states']
    assert c['train'][0]['raw_std_6'] == pytest.approx([.980665]*6)
    with pytest.raises(ValueError): convert(c)


def test_pre_unit_fix_calibration_is_rejected():
    with pytest.raises(ValueError, match='N/Nm'):
        ForceFeedback({'status':'validated_current_mount'}, np.eye(4), Settings())
