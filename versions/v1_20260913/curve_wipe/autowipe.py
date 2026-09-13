"""Automatic scan-to-plan workflow. No robot motion; candidates are not authorization."""
import argparse
from datetime import datetime
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .capture import capture
from .execute import ExecutionError, prepare_plan, ee_position, STANDOFF
from .geometry import segment_red
from .plan import ROOT, build_plan


def detect_regions(bgr, xyz, min_pixels=80, padding=10):
    """Find all red components with real depth; never fill gaps in surface data."""
    if min_pixels < 1 or padding < 0:
        raise ValueError('invalid detector settings')
    red = segment_red(bgr, np.ones(bgr.shape[:2], dtype=bool))
    valid = np.isfinite(xyz).all(axis=2) & (xyz[..., 2] > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype('uint8'), 8)
    regions = []
    height, width = red.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        depth_pixels = int(np.count_nonzero((labels == label) & valid))
        if area < min_pixels or depth_pixels < min_pixels:
            continue
        x0, y0 = max(0, x-padding), max(0, y-padding)
        x1, y1 = min(width, x+w+padding), min(height, y+h+padding)
        regions.append(dict(roi_xywh=[x0, y0, x1-x0, y1-y0],
                            red_pixels=area, valid_depth_pixels=depth_pixels))
    return sorted(regions, key=lambda r: (-r['valid_depth_pixels'], r['roi_xywh']))


def approach_proposal(prepared):
    """Fixed-attitude lift/traverse/descend proposal in the observed local half-space.

    The plane clearance applies only to TCP, not the arm or tool footprint.
    No maximum travel distance: IK, scene collision and timing remain unchecked.
    """
    p = prepared
    start = p['capture'][:3, 3]
    normal = p['normal']
    tcp = start + p['rotation'] @ p['tcp_offset']
    clearance = float(np.min((tcp-p['points']) @ normal))
    if clearance < .020:
        raise ExecutionError('start TCP has less than 20 mm local clearance')
    entry = ee_position(p['points'][0], STANDOFF, p['rotation'], p['tcp_offset'], normal)
    entry_tcp = entry + p['rotation'] @ p['tcp_offset']
    end_clearance = float(np.min((entry_tcp-p['points']) @ normal))
    if end_clearance < .020:
        raise ExecutionError('precontact TCP has less than 20 mm local clearance')
    transit_clearance = max(.100, clearance, end_clearance)
    raised_start = start + (transit_clearance-clearance)*normal
    raised_entry = entry + (transit_clearance-end_clearance)*normal
    points = [start, raised_start, raised_entry, entry]
    poses = []
    for position in points:
        pose = p['capture'].copy()
        pose[:3, 3] = position
        if not poses or np.linalg.norm(position-np.array(poses[-1])[:3, 3]) > 1e-8:
            poses.append(pose.tolist())
    length = sum(np.linalg.norm(np.array(b)[:3, 3]-np.array(a)[:3, 3])
                 for a, b in zip(poses, poses[1:]))
    return dict(poses_base_ee=poses, path_length_m=float(length),
                direct_entry_distance_m=float(np.linalg.norm(entry-start)),
                minimum_local_tcp_clearance_m=min(clearance, end_clearance),
                transit_local_tcp_clearance_m=transit_clearance,
                collision_checked=False, reachability_checked=False,
                executable=False, rescan_after_approach_required=True)


def analyze(snapshot, output, *, handeye=None, tcp=None, min_pixels=80):
    """Analyze a saved snapshot without importing or connecting robot control."""
    snapshot, output = Path(snapshot), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(snapshot, allow_pickle=False) as data:
        bgr, xyz = data['bgr'].copy(), data['xyz_rgb_m'].copy()
    regions = detect_regions(bgr, xyz, min_pixels)
    report = dict(status='no_target', motion_started=False, executable=False,
                  snapshot=str(snapshot.resolve()), regions=regions, candidates=[], rejected=[],
                  target_semantics='red color only; not a verified stain classifier',
                  selected_candidate=None)
    overlay = bgr.copy()
    for index, region in enumerate(regions):
        x, y, w, h = region['roi_xywh']
        cv2.rectangle(overlay, (x, y), (x+w-1, y+h-1), (0, 255, 255), 2)
        cv2.putText(overlay, str(index), (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (0, 255, 255), 2)
        try:
            plan = build_plan(snapshot, output/f'region_{index:03d}', region['roi_xywh'],
                              handeye or ROOT/'config/handeye_result.json',
                              tcp or ROOT/'config/eraser_tcp_candidate.json', preview_only=True)
            # Auto-detection is not a user confirmation or permission to move.
            plan['metadata'].update(calibration_mount_unchanged_user_confirmed=False,
                                    executable_candidate=False, execution_mode='autowipe_preview')
            plan_file = output/f'region_{index:03d}'/'plan.json'
            plan_file.write_text(json.dumps(plan, indent=2, allow_nan=False))
        except (ValueError, KeyError) as exc:
            report['rejected'].append(dict(region=index, reason=str(exc)))
            continue
        for segment in range(len(plan['segments'])):
            try:
                prepared = prepare_plan(plan, segment)
                approach = approach_proposal(prepared)
                report['candidates'].append(dict(region=index, segment=segment,
                    plan=str(plan_file.resolve()), wipe_length_m=prepared['length_m'],
                    max_normal_angle_deg=prepared['max_normal_angle_deg'], approach=approach))
            except (ExecutionError, ValueError) as exc:
                report['rejected'].append(dict(region=index, segment=segment, reason=str(exc)))
    if report['candidates']:
        # Prefer more wipe per segment, then shorter transit; never imply complete coverage.
        report['candidates'].sort(key=lambda c: (-c['wipe_length_m'], c['approach']['path_length_m']))
        report['selected_candidate'] = 0
        report['status'] = 'awaiting_motion_planning'
        report['blockers'] = ['target_workspace_not_verified', 'whole_arm_ik_not_checked',
                              'tool_and_environment_collision_not_checked',
                              'post_approach_rescan_and_force_precheck_required']
    elif regions:
        report['status'] = 'no_fixed_orientation_candidate'
    report['coverage_guaranteed'] = False
    cv2.imwrite(str(output/'targets.png'), overlay)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, help='replay saved data without hardware access')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ip', default='172.16.0.2')
    parser.add_argument('--serial', default='243722072895')
    parser.add_argument('--min-pixels', type=int, default=80)
    parser.add_argument('--motion-config', type=Path, help='also run a read-only MoveIt scene audit')
    parser.add_argument('--depth-scene', action='store_true',
                        help='build/apply observed RealSense voxels and audit with MoveIt; no environment CAD or motion')
    args = parser.parse_args()
    if args.depth_scene and args.motion_config:
        parser.error('--depth-scene generates its own motion config; do not combine with --motion-config')
    output = args.output or ROOT/'output'/('autowipe_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = args.snapshot
        if snapshot is None:
            capture(output/'capture', args.serial, args.ip)
            snapshot = output/'capture'/'snapshot.npz'
        report = analyze(snapshot, output, min_pixels=args.min_pixels)
        if args.depth_scene and report['candidates']:
            scene_dir = output/'depth_scene'
            subprocess.run([str(ROOT/'run.sh'), 'depth_scene', '--snapshot', str(Path(snapshot).resolve()),
                            '--output', str(scene_dir.resolve()), '--apply'], check=True, timeout=90)
            args.motion_config = scene_dir/'motion_config.json'
            report['depth_scene'] = json.loads((scene_dir/'apply_result.json').read_text())
        if args.motion_config is not None and report['candidates']:
            audit_file = output/'motion_audit.json'
            subprocess.run([str(ROOT/'run.sh'), 'check_approach',
                            '--report', str((output/'report.json').resolve()),
                            '--config', str(args.motion_config.resolve()),
                            '--output', str(audit_file.resolve())],
                           check=False, timeout=55)
            report['motion_audit'] = json.loads(audit_file.read_text())
            report['status'] = ('awaiting_execution_integration'
                                if report['motion_audit']['status'] == 'sampled_scene_audit_passed'
                                else 'motion_planning_blocked')
            # Audit output cannot authorize motion or turn preview plans executable.
            (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    except Exception as exc:
        report = dict(status='failed', motion_started=False, executable=False,
                      error=str(exc), error_type=type(exc).__name__)
        (output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report['status'], motion_started=False,
                         candidates=len(report.get('candidates', [])),
                         report=str(output/'report.json')), indent=2))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
