"""Find red marks, sample a local curved surface, and save a short-path preview."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import plan_surface, segment_red, _transform

ROOT = Path(__file__).resolve().parents[1]


def resolve_tool_frames(tcp_data):
    """Resolve the gripper reference and the active eraser contact point.

    The legacy calibration stores the eraser contact *centre* in
    ``T_EE_L_TCP_eraser``.  New calibrations may provide an explicit front
    contact transform, or a signed longitudinal distance from the gripper
    centre.  Keeping these frames separate prevents a path generated for the
    front edge from silently being executed with the eraser centre.
    """
    t = tcp_data
    legacy = _transform(t['T_EE_L_TCP_eraser'], 'T_EE_tcp_legacy_contact_center')

    grip_raw = t.get('T_EE_L_GRIP_CENTER')
    if grip_raw is not None:
        grip = _transform(grip_raw, 'T_EE_grip_center')
    else:
        # The existing candidate records the axial distance to the gripper
        # centre.  Its XY origin is the taught gripper centre.
        dims = t.get('dimensions_mm', {})
        axial_mm = dims.get('EE_to_grip_center_axial')
        if axial_mm is None:
            grip = legacy.copy()
            grip[:3, 3] = 0.0
        else:
            grip = legacy.copy()
            grip[:3, 3] = legacy[:3, :3] @ np.array([0.0, 0.0, float(axial_mm) / 1000.0])

    requested = t.get('active_contact_reference', 'legacy_contact_center')
    wipe_raw = t.get('T_EE_L_WIPE_CENTER')
    front_raw = t.get('T_EE_L_ERASER_FRONT')
    # An explicit wipe-centre matrix has priority for wipe_center.  An
    # explicit eraser-front matrix or the compact signed calibration has
    # priority for eraser_front, so the two references can coexist.
    if requested == 'wipe_center' and wipe_raw is not None:
        front = _transform(wipe_raw, 'T_EE_wipe_center')
        front_calibrated = True
    elif requested == 'eraser_front' and front_raw is not None:
        front = _transform(front_raw, 'T_EE_eraser_front')
        front_calibrated = True
    else:
        # Optional compact calibration for a front edge measured along the
        # eraser long axis.  Positive X follows the convention documented in
        # eraser_tcp_candidate.json; the user can therefore enter a negative
        # value for the visible/front end.
        fc = t.get('front_contact_calibration', {})
        longitudinal_mm = fc.get('grip_center_to_front_longitudinal_mm')
        if longitudinal_mm is None:
            front = None
            front_calibrated = False
        else:
            axial_mm = fc.get('grip_center_to_front_axial_mm',
                              t.get('dimensions_mm', {}).get('grip_center_to_pad_axial', 10.0))
            front = grip.copy()
            front[:3, 3] = (grip[:3, 3] + grip[:3, :3] @
                            np.array([float(longitudinal_mm) / 1000.0,
                                      0.0, float(axial_mm) / 1000.0]))
            front_calibrated = True
    if requested in ('eraser_front', 'wipe_center'):
        if front is None:
            raise ValueError(
                f'active_contact_reference={requested} requires '
                'T_EE_L_WIPE_CENTER, T_EE_L_ERASER_FRONT, or front_contact_calibration.'
            )
        contact, name = front, requested
    elif requested == 'legacy_contact_center':
        contact, name = legacy, 'legacy_contact_center'
    else:
        raise ValueError('active_contact_reference must be legacy_contact_center, wipe_center, or eraser_front')
    return dict(grip=grip, legacy=legacy, front=front, contact=contact,
                name=name, front_calibrated=front_calibrated)


def load_calibration(handeye, tcp, *, contact_reference=None):
    h, t = json.loads(Path(handeye).read_text()), json.loads(Path(tcp).read_text())
    Tec = _transform(h['T_EE_L_C_LRGB'], 'T_EE_camera')
    if contact_reference is not None:
        t = dict(t, active_contact_reference=contact_reference)
    Tet = resolve_tool_frames(t)['contact']
    if not np.allclose(Tec @ np.asarray(h['T_C_LRGB_EE_L']), np.eye(4), atol=1e-5):
        raise ValueError('handeye inverse is inconsistent')
    return h, t, Tec, Tet


def make_roi(shape, rectangle):
    x, y, w, h = map(int, rectangle)
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > shape[1] or y + h > shape[0]:
        raise ValueError('ROI must be entirely inside image: x y width height')
    roi = np.zeros(shape, dtype=bool)
    roi[y:y+h, x:x+w] = True
    return roi


def build_plan(snapshot, output, rectangle, handeye, tcp, *, preview_only=False):
    data = np.load(snapshot, allow_pickle=False)
    bgr, xyz = data['bgr'], data['xyz_rgb_m']
    meta = json.loads(str(data['metadata_json']))
    h, t, Tec, Tet = load_calibration(handeye, tcp)
    frames = resolve_tool_frames(t)
    if meta.get('camera_serial') != h['camera_serial']:
        raise ValueError('camera serial differs from handeye calibration')
    if meta.get('xyz_frame') != 'C_LRGB' or meta.get('xyz_units') != 'metres':
        raise ValueError('expected RGB optical-frame XYZ in metres')
    if not meta.get('static_pose_validated'):
        raise ValueError('snapshot has no validated static robot pose; recapture with --robot-ip')
    Tbe = _transform(meta['T_base_ee'], 'T_base_ee')
    roi = make_roi(bgr.shape[:2], rectangle)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    mask = segment_red(bgr, roi)
    cv2.imwrite(str(out / 'red_mask.png'), mask.astype(np.uint8) * 255)
    try:
        plan = plan_surface(bgr, xyz, roi, Tbe @ Tec, Tet)
    except ValueError as exc:
        (out / 'detection_report.json').write_text(json.dumps({
            'status': 'no_plan', 'reason': str(exc), 'red_pixels': int(mask.sum()),
            'valid_depth_pixels_in_roi': int((roi & np.isfinite(xyz).all(axis=2)).sum())}, indent=2))
        raise
    # Preserve the legacy fixed-pose fields for comparison.  The executor now
    # uses each waypoint's surface-normal TCP frame when those fields exist.
    n0 = -Tbe[:3, 2]
    for segment in plan['segments']:
        for p in segment['waypoints']:
            surface = np.asarray(p['surface_point_base_m'])
            contact = Tbe.copy()
            contact[:3, 3] = surface - Tbe[:3, :3] @ Tet[:3, 3]
            stand = contact.copy()
            stand[:3, 3] += .05 * n0
            p['T_base_ee_fixed_contact'] = contact.tolist()
            p['T_base_ee_fixed_standoff'] = stand.tolist()
    synthetic = bool(meta.get('synthetic', False))
    plan['metadata'].update(
        T_base_ee_capture=Tbe.tolist(), T_ee_tcp=Tet.tolist(),
        T_ee_grip_center=frames['grip'].tolist(),
        contact_reference=frames['name'],
        front_contact_calibrated=frames['front_calibrated'],
        F_T_EE_capture=np.asarray(meta['robot_before']['F_T_EE']).reshape(4, 4, order='F').tolist(),
        synthetic=synthetic, executable_candidate=not synthetic and not preview_only,
        offline_preview_only=True, execution_mode='surface_normal_aligned',
        camera_serial=h['camera_serial'], snapshot=str(Path(snapshot).resolve()),
        roi_xywh=list(rectangle), handeye_status=h.get('status'), tcp_status=t.get('status'),
        calibration_mount_unchanged_user_confirmed=False if preview_only else True,
        handeye_sha256=hashlib.sha256(Path(handeye).read_bytes()).hexdigest(),
        tcp_sha256=hashlib.sha256(Path(tcp).read_bytes()).hexdigest(),
        execution_note='Candidate only. Executor follows each waypoint surface normal, checks actual pose, and uses one <=200mm segment; no automatic between-segment travel.')
    (out / 'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False))
    overlay = bgr.copy()
    overlay[mask] = (.4 * overlay[mask] + .6 * np.array([0, 255, 255])).astype(np.uint8)
    for i, seg in enumerate(plan['segments']):
        uv = np.rint([p['pixel_uv'] for p in seg['waypoints']]).astype(np.int32)
        cv2.polylines(overlay, [uv], False, (0, 255, 0), 1)
        if len(uv):
            cv2.putText(overlay, str(i), tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0), 1)
    x, y, w, hh = rectangle
    cv2.rectangle(overlay, (x, y), (x+w-1, y+hh-1), (255, 255, 0), 1)
    cv2.imwrite(str(out / 'overlay.png'), overlay)
    plot_preview(plan, xyz, roi, Tbe @ Tec, out / 'trajectory.png')
    print(json.dumps({'plan': str(out/'plan.json'), 'segments': len(plan['segments']),
                      'red_pixels': int(mask.sum()), 'synthetic': synthetic}, indent=2))
    return plan


def plot_preview(plan, xyz, roi, Tbc, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = roi & np.isfinite(xyz).all(axis=2)
    points = xyz[valid][::max(1, int(valid.sum()) // 5000)]
    points = points @ Tbc[:3, :3].T + Tbc[:3, 3]
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T, s=1, c='gray', alpha=.2)
    for i, segment in enumerate(plan['segments']):
        p = np.array([v['surface_point_base_m'] for v in segment['waypoints']])
        ax.plot(*p.T, linewidth=1.5)
    ax.set(xlabel='Base X / m', ylabel='Base Y / m', zlabel='Base Z / m',
           title=('SYNTHETIC TEST - ' if plan['metadata']['synthetic'] else '') + 'Red-mark surface paths (preview)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def synthetic_snapshot(output):
    """Analytic curved patch for offline checks; explicitly forbidden on hardware."""
    h = json.loads((ROOT/'config/handeye_result.json').read_text())
    Tec = np.asarray(h['T_EE_L_C_LRGB'])
    Tbe = np.diag([1., -1., -1., 1.])
    Tbe[:3, 3] = [.40, .10, .50]
    Tbc = Tbe @ Tec
    yy, xx = np.mgrid[:121, :161]
    x, y = (xx-80)*.0005, (yy-60)*.0005
    base = np.stack([.40+x, .10+y, .12 + 1.2*x*x + .3*y*y], axis=-1)
    xyz = (base - Tbc[:3, 3]) @ Tbc[:3, :3]
    bgr = np.full((*xx.shape, 3), 225, dtype=np.uint8)
    bgr[(abs(y) < .006) & (abs(x) < .032)] = [25, 25, 210]
    xyz[:, 78:83] = np.nan  # Real holes must split paths, never connect across them.
    Ftee = np.eye(4)
    Ftee[2, 3] = .1034
    meta = dict(source='analytic_synthetic', synthetic=True, camera_serial=h['camera_serial'],
                xyz_frame='C_LRGB', xyz_units='metres', static_pose_validated=True,
                T_base_ee=Tbe.tolist(), robot_before={'F_T_EE':Ftee.flatten(order='F').tolist()})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'snapshot.npz', bgr=bgr, xyz_rgb_m=xyz, metadata_json=json.dumps(meta))
    cv2.imwrite(str(output/'color.png'), bgr)
    return output/'snapshot.npz'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--roi', type=int, nargs=4, metavar=('X','Y','WIDTH','HEIGHT'))
    p.add_argument('--handeye', default=str(ROOT/'config/handeye_result.json'))
    p.add_argument('--tcp', default=str(ROOT/'config/eraser_tcp_candidate.json'))
    a = p.parse_args()
    if a.synthetic:
        snapshot = synthetic_snapshot(Path(a.output)/'synthetic_input')
        roi = a.roi or [5, 5, 151, 111]
    else:
        if not a.snapshot or not a.roi:
            p.error('real images require --snapshot and explicit --roi X Y WIDTH HEIGHT')
        snapshot, roi = a.snapshot, a.roi
    build_plan(snapshot, a.output, roi, a.handeye, a.tcp)


if __name__ == '__main__':
    main()
