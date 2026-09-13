import json
from pathlib import Path

import numpy as np
import pytest

from curve_wipe.capture import rasterize_xyz
from curve_wipe.plan import load_calibration, make_roi, synthetic_snapshot, build_plan, ROOT, resolve_tool_frames


def test_sdk_raster_uses_rgb_extrinsics_and_nearest_point():
    T = np.eye(4)
    T[0, 3] = .025
    xyz = rasterize_xyz([[0,0,2], [0,0,1], [0,0,0]], [[.5,.5]]*3, T, 2, 2)
    np.testing.assert_allclose(xyz[1,1], [.025,0,1])
    assert np.isnan(xyz[0,0]).all()


def test_calibration_inverse_and_tcp_offset():
    _, _, Tec, Tet = load_calibration(ROOT/'config/handeye_result.json', ROOT/'config/eraser_tcp_candidate.json')
    assert .060 < np.linalg.norm(Tet[:3,3]) < .063
    assert np.linalg.det(Tec[:3,:3]) == pytest.approx(1)


def test_tool_frames_keep_gripper_and_front_contact_separate():
    tcp = json.loads((ROOT/'config/eraser_tcp_candidate.json').read_text())
    frames = resolve_tool_frames(tcp)
    assert frames['name'] == 'wipe_center'
    assert frames['front'] is not None
    assert frames['grip'][2, 3] == pytest.approx(.0466)
    assert frames['contact'][0, 3] == pytest.approx(0.0)
    assert frames['contact'][2, 3] == pytest.approx(.0616)


def test_front_contact_can_be_activated_from_signed_offset():
    tcp = json.loads((ROOT/'config/eraser_tcp_candidate.json').read_text())
    tcp['active_contact_reference'] = 'eraser_front'
    tcp['front_contact_calibration']['grip_center_to_front_longitudinal_mm'] = -23.0
    tcp['front_contact_calibration']['grip_center_to_front_axial_mm'] = 10.0
    frames = resolve_tool_frames(tcp)
    assert frames['name'] == 'eraser_front'
    assert frames['front_calibrated'] is True
    assert frames['front'][2, 3] == pytest.approx(.0566)


def test_invalid_roi_rejected():
    with pytest.raises(ValueError):
        make_roi((100,100), [-1,0,20,20])


def test_synthetic_complete_pipeline_not_executable(tmp_path):
    snap = synthetic_snapshot(tmp_path/'input')
    plan = build_plan(snap, tmp_path/'out', [5,5,151,111],
                      ROOT/'config/handeye_result.json', ROOT/'config/eraser_tcp_candidate.json')
    assert plan['metadata']['synthetic'] is True
    assert plan['metadata']['executable_candidate'] is False
    assert (tmp_path/'out/trajectory.png').is_file()
    assert len(plan['segments']) >= 2
    # Fixed-orientation execution geometry includes the full lateral TCP offset.
    Tbe = np.array(plan['metadata']['T_base_ee_capture'])
    Tet = np.array(plan['metadata']['T_ee_tcp'])
    p = plan['segments'][0]['waypoints'][0]
    actual_tcp = np.array(p['T_base_ee_fixed_contact']) @ Tet
    np.testing.assert_allclose(actual_tcp[:3,3], p['surface_point_base_m'], atol=1e-9)
