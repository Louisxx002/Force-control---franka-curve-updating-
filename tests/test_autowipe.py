import json
import numpy as np
import pytest

from curve_wipe.autowipe import detect_regions, approach_proposal, analyze
from curve_wipe.execute import ExecutionError
from curve_wipe.plan import synthetic_snapshot


def test_detector_ignores_tiny_and_missing_depth_targets():
    bgr = np.zeros((60, 100, 3), dtype=np.uint8)
    xyz = np.ones((60, 100, 3), dtype=float)
    bgr[10:20, 10:30, 2] = 255
    bgr[30:50, 60:80, 2] = 255
    xyz[30:50, 60:80] = np.nan
    bgr[1, 1, 2] = 255
    regions = detect_regions(bgr, xyz)
    assert len(regions) == 1
    assert regions[0]['valid_depth_pixels'] == 200
    assert regions[0]['roi_xywh'] == [0, 0, 40, 30]


def prepared(start=(.5, 0, .15)):
    pose = np.eye(4)
    pose[:3, 3] = start
    return dict(capture=pose, normal=np.array([0., 0, 1]),
                rotation=np.eye(3), tcp_offset=np.array([.04, 0, .02]),
                points=np.array([[0., 0, 0], [.01, 0, 0]]))


def test_long_approach_is_preview_only_and_accounts_for_tool_offset():
    proposal = approach_proposal(prepared())
    assert proposal['direct_entry_distance_m'] > .5
    assert not proposal['executable']
    assert not proposal['reachability_checked']
    poses = np.array(proposal['poses_base_ee'])
    np.testing.assert_allclose(poses[-1, :3, 3] + [.04, 0, .02], [0, 0, .05])
    assert proposal['minimum_local_tcp_clearance_m'] == pytest.approx(.05)
    for a, b in zip(poses, poses[1:]):
        for u in np.linspace(0, 1, 21):
            position = (1-u)*a[:3, 3]+u*b[:3, 3]
            assert position[2]+.02 >= .05-1e-10


def test_contact_start_is_rejected():
    with pytest.raises(ExecutionError, match='clearance'):
        approach_proposal(prepared((0, 0, -.01)))


def test_offline_pipeline_never_marks_auto_targets_executable(tmp_path):
    snapshot = synthetic_snapshot(tmp_path/'input')
    report = analyze(snapshot, tmp_path/'result', min_pixels=5)
    assert report['regions']
    assert report['motion_started'] is False
    assert report['executable'] is False
    assert (tmp_path/'result'/'targets.png').exists()
    plans = list((tmp_path/'result').glob('region_*/plan.json'))
    assert plans
    for path in plans:
        meta = json.loads(path.read_text())['metadata']
        assert meta['executable_candidate'] is False
        assert meta['calibration_mount_unchanged_user_confirmed'] is False
