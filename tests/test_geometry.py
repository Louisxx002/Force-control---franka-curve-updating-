import json

import numpy as np
import pytest

from curve_wipe.geometry import TARGET_COLORS, plan_surface, segment_red, segment_target


def scene(shape=(51, 81), curvature=3.0):
    v, u = np.indices(shape)
    x = (u - shape[1] / 2) * 0.001
    y = (v - shape[0] / 2) * 0.001
    z = 0.5 + curvature * x ** 2
    xyz = np.stack((x, y, z), axis=-1)
    bgr = np.zeros((*shape, 3), dtype=np.uint8)
    bgr[10:-10, 10:-10, 2] = 255
    return bgr, xyz, np.ones(shape, dtype=bool)


def planned(bgr, xyz, roi, **config):
    return plan_surface(bgr, xyz, roi, np.eye(4), np.eye(4), config)


def test_red_mask_respects_roi_and_both_hue_ranges():
    import cv2
    hsv = np.array([[[0, 255, 255], [179, 255, 255], [60, 255, 255], [0, 255, 255]]], np.uint8)
    roi = np.array([[True, True, True, False]])
    mask = segment_red(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), roi)
    assert mask.tolist() == [[True, True, False, False]]
    with pytest.raises(ValueError, match="explicit"):
        segment_red(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), None)


def test_common_color_targets_share_the_mask_interface():
    import cv2
    roi = np.ones((80, 220), dtype=bool)
    hues = {
        'orange': 12, 'yellow': 30, 'lime': 50, 'green': 70,
        'cyan': 95, 'blue': 115, 'violet': 140, 'purple': 155,
        'magenta': 170, 'pink': 165, 'brown': 15,
    }
    for color in TARGET_COLORS:
        if color in ('red', 'white', 'black'):
            continue
        hsv = np.zeros((80, 220, 3), dtype=np.uint8)
        hsv[30:42, 20:200] = ([0, 0, 150] if color == 'gray'
                               else [hues[color], 220, 180])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        mask = segment_target(bgr, roi, color)
        assert mask[35, 100] and int(mask.sum()) > 1000, color


def test_local_normals_follow_analytic_curved_surface():
    bgr, xyz, roi = scene()
    result = planned(bgr, xyz, roi)
    angles = []
    for segment in result["segments"]:
        for waypoint in segment["waypoints"]:
            p = np.array(waypoint["surface_point_base_m"])
            expected = np.array([6 * p[0], 0, -1])
            expected /= np.linalg.norm(expected)
            n = np.array(waypoint["normal_out_base"])
            angles.append(np.degrees(np.arccos(np.clip(n @ expected, -1, 1))))
            assert n @ -p > 0
    assert max(angles) < 0.1
    assert result["metadata"]["offline_preview_only"] is True
    json.dumps(result, allow_nan=False)


def test_nan_depth_hole_is_never_bridged():
    bgr, xyz, roi = scene(curvature=0)
    xyz[:, 37:44] = np.nan
    result = planned(bgr, xyz, roi)
    assert len(result["segments"]) >= 2
    for segment in result["segments"]:
        us = [wp["pixel_uv"][0] for wp in segment["waypoints"]]
        assert max(us) < 37 or min(us) >= 44


def test_depth_jump_is_split_or_rejected_without_bridging():
    bgr, xyz, roi = scene(curvature=0)
    xyz[:, 41:, 2] += 0.06
    result = planned(bgr, xyz, roi)
    for segment in result["segments"]:
        us = [wp["pixel_uv"][0] for wp in segment["waypoints"]]
        assert max(us) < 41 or min(us) >= 41


def test_tcp_ee_transform_and_outward_standoff():
    bgr, xyz, roi = scene()
    T_bc, T_et = np.eye(4), np.eye(4)
    T_bc[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T_bc[:3, 3] = [0.2, -0.1, 0.3]
    T_et[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T_et[:3, 3] = [0.01, 0.02, 0.15]
    result = plan_surface(bgr, xyz, roi, T_bc, T_et, {})
    for segment in result["segments"]:
        for wp in segment["waypoints"]:
            tcp = np.array(wp["T_base_tcp_contact"])
            ee = np.array(wp["T_base_ee_contact"])
            stand = np.array(wp["T_base_tcp_standoff"])
            stand_ee = np.array(wp["T_base_ee_standoff"])
            np.testing.assert_allclose(ee @ T_et, tcp, atol=1e-12)
            np.testing.assert_allclose(stand_ee @ T_et, stand, atol=1e-12)
            np.testing.assert_allclose(stand[:3, 3] - tcp[:3, 3], np.array(wp["normal_out_base"]) * 0.05, atol=1e-12)


def test_frame_orthonormal_and_scan_direction_does_not_reverse():
    bgr, xyz, roi = scene()
    result = planned(bgr, xyz, roi)
    traversal = []
    for segment in result["segments"]:
        points = segment["waypoints"]
        traversal.append(np.sign(points[-1]["pixel_uv"][0] - points[0]["pixel_uv"][0]))
        for wp in points:
            R = np.array(wp["T_base_tcp_contact"])[:3, :3]
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-12)
            np.testing.assert_allclose(R[:, 2], -np.array(wp["normal_out_base"]), atol=1e-12)
            assert R[0, 0] > 0.97
    assert traversal[:2] == [1, -1]


def test_sampling_uses_physical_arc_length():
    bgr, xyz, roi = scene()
    result = planned(bgr, xyz, roi, sample_step_m=0.003)
    for segment in result["segments"]:
        points = np.array([wp["surface_point_base_m"] for wp in segment["waypoints"]])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert np.max(distances) <= 0.003000001
        # The short endpoint interval comes first on reversed raster rows.
        np.testing.assert_allclose(np.sort(distances)[1:], 0.003, atol=1e-7)


def test_empty_target_and_line_degeneracy_rejected():
    bgr, xyz, roi = scene()
    with pytest.raises(ValueError, match="no red target"):
        planned(bgr * 0, xyz, roi)
    roi[:] = False
    roi[25] = True
    with pytest.raises(ValueError, match="no acceptable"):
        planned(bgr, xyz, roi, min_neighbors=5)


def test_high_slope_rejected():
    bgr, xyz, roi = scene(curvature=0)
    xyz[..., 2] += 2 * xyz[..., 0]
    with pytest.raises(ValueError, match="no acceptable"):
        planned(bgr, xyz, roi, max_view_angle_deg=35)
