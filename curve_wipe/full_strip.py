"""Plan the currently visible extent of a target strip.

Preview only. The selected connected component is the complete target for this
scan; material hidden behind the tool or outside the image is intentionally not
inferred. Missing depth or surface discontinuities still block the visible plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .geometry import segment_target, _local_geometry, _resample, _unit, _transform, DEFAULT_CONFIG
from .plan import ROOT, load_calibration, make_roi, plot_preview, resolve_tool_frames
from .execute import prepare_plan, ExecutionError


WHITE_FILL_RADIUS_PX = 18
WHITE_FILL_MAX_RMS_M = 0.003
WHITE_CENTERLINE_SMOOTH_PASSES = 3


def _fill_white_centerline(xyz, valid, target, columns, rows):
    """Estimate reflective-tape holes from nearby non-tape surface points.

    The tape itself is excluded from each fit.  A fit is accepted only when it
    has enough neighbors and sub-3 mm RMS residual; otherwise the lane remains
    invalid and the full-strip admission check rejects execution.
    """
    filled = 0
    residuals = []
    for u, v in zip(columns, rows):
        u, v = int(round(u)), int(round(v))
        radius = WHITE_FILL_RADIUS_PX
        y0, y1 = max(0, v-radius), min(xyz.shape[0], v+radius+1)
        x0, x1 = max(0, u-radius), min(xyz.shape[1], u+radius+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        use = valid[y0:y1, x0:x1] & ~target[y0:y1, x0:x1]
        points = xyz[y0:y1, x0:x1][use]
        design = np.column_stack((xx[use]-u, yy[use]-v, np.ones(int(use.sum()))))
        if len(points) < 20:
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(design, points, rcond=None)
        if rank < 3:
            continue
        residual = np.linalg.norm(points - design @ coefficients, axis=1)
        rms = float(np.sqrt(np.mean(residual**2)))
        if not np.isfinite(rms) or rms > WHITE_FILL_MAX_RMS_M:
            continue
        xyz[v, u] = np.array([0.0, 0.0, 1.0]) @ coefficients
        valid[v, u] = True
        filled += 1
        residuals.append(rms)
    if filled:
        # Remove pixel-scale depth jitter from the reconstructed centreline
        # while preserving its measured end points.
        sequence = np.array([xyz[int(round(v)), int(round(u))]
                             for u, v in zip(columns, rows)])
        for _ in range(WHITE_CENTERLINE_SMOOTH_PASSES):
            sequence[1:-1] = (sequence[:-2] + 2*sequence[1:-1] + sequence[2:]) / 4
        for (u, v), point in zip(zip(columns, rows), sequence):
            xyz[int(round(v)), int(round(u))] = point
    return filled, residuals


def _principal_strip_pixels(target):
    """Return center/edge pixels for an elongated strip at any image angle."""
    yy, xx = np.where(target)
    pixels = np.column_stack((xx, yy)).astype(float)
    center = pixels.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(pixels.T))
    axis = vectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    transverse = np.array([-axis[1], axis[0]])
    longitudinal = (pixels-center) @ axis
    lateral = (pixels-center) @ transverse
    bins = np.arange(longitudinal.min(), longitudinal.max()+1.0, 2.0)
    centers, lows, highs = [], [], []
    for start in bins:
        use = (longitudinal >= start) & (longitudinal < start+2.0)
        if not np.any(use):
            continue
        s = float(np.mean(longitudinal[use]))
        lo, hi = float(np.min(lateral[use])), float(np.max(lateral[use]))
        base = center + s*axis
        centers.append(base + .5*(lo+hi)*transverse)
        lows.append(base + lo*transverse)
        highs.append(base + hi*transverse)
    if len(centers) < 3:
        raise ValueError('black target has too few longitudinal samples')
    return np.asarray(centers), np.asarray(lows), np.asarray(highs)


def strip_envelope(bgr, roi, target_color='red'):
    target = segment_target(bgr, roi, target_color, allow_border=True)
    contours, _ = cv2.findContours(target.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 80]
    if len(contours) != 1:
        raise ValueError(f'select an ROI containing exactly one connected {target_color} strip')
    contour = contours[0]
    x, y, width, height = cv2.boundingRect(contour)
    principal_geometry = target_color in ('black', 'white')
    if not principal_geometry and width < 3*height:
        raise ValueError('this planner requires a nearly horizontal elongated strip')
    # Filled external outline identifies the same object across printed letters.
    # It never changes XYZ or marks invalid depth as measured.
    filled = np.zeros(target.shape, np.uint8)
    cv2.drawContours(filled, [contour], -1, 1, -1)
    if principal_geometry:
        return target, filled.astype(bool), *_principal_strip_pixels(filled.astype(bool))
    low, high = [], []
    envelope = np.zeros_like(target)
    for u in range(x, x+width):
        rows = np.flatnonzero(filled[:, u])
        if not len(rows): raise ValueError('target has a disconnected longitudinal column')
        lo, hi = int(rows[0]), int(rows[-1])
        if not roi[lo:hi+1, u].all(): raise ValueError('strip envelope exceeds selected ROI')
        low.append(lo); high.append(hi); envelope[lo:hi+1, u] = True
    return target, envelope, np.arange(x, x+width), np.array(low), np.array(high)


def plan_strip(bgr, xyz, roi, tbc, tet, lane_count=1, tgrip=None, target_color='red'):
    if not isinstance(lane_count, int) or not 1 <= lane_count <= 25:
        raise ValueError('lane_count must be an integer between 1 and 25')
    target, envelope, columns, low, high = strip_envelope(bgr, roi, target_color)
    principal_geometry = target_color in ('black', 'white')
    xyz = np.asarray(xyz, float)
    if xyz.shape != (*target.shape, 3): raise ValueError('XYZ image shape mismatch')
    tbc, tet = _transform(tbc, 'T_base_camera'), _transform(tet, 'T_ee_tcp')
    if tgrip is not None:
        tgrip = _transform(tgrip, 'T_ee_grip_center')
    valid = np.asarray(roi) & np.isfinite(xyz).all(-1) & (xyz[...,2] > 0)
    cfg = dict(DEFAULT_CONFIG)
    if target_color in ('white', 'black'):
        cfg.update(neighborhood_radius_px=WHITE_FILL_RADIUS_PX,
                   max_neighbor_distance_m=0.040)
    segments, failures, dense_lanes = [], [], []
    white_filled_points = 0
    white_fill_residuals = []
    if principal_geometry:
        center_pixels, low_pixels, high_pixels = _principal_strip_pixels(target)
    for lane, fraction in enumerate([.5] if lane_count == 1 else np.linspace(0, 1, lane_count)):
        if principal_geometry:
            pixel_points = low_pixels*(1-fraction) + high_pixels*fraction
        else:
            rows = np.rint(low+fraction*(high-low)).astype(int)
            pixel_points = np.column_stack((columns, rows)).astype(float)
        if target_color in ('white', 'black'):
            count, residuals = _fill_white_centerline(
                xyz, valid, target, pixel_points[:, 0], pixel_points[:, 1])
            white_filled_points += count
            white_fill_residuals.extend(residuals)
        fragment = []; lane_errors = []
        for u, v in pixel_points:
            u, v = int(round(u)), int(round(v))
            if not valid[v,u]:
                lane_errors.append(dict(pixel_uv=[int(u),int(v)],reason='missing_depth')); continue
            item, reason = _local_geometry(xyz, valid, int(u), int(v), cfg)
            if reason:
                lane_errors.append(dict(pixel_uv=[int(u),int(v)],reason=reason)); continue
            if fragment:
                a = fragment[-1]
                if (np.linalg.norm(item['point']-a['point']) > cfg['max_depth_jump_m'] or
                    item['normal']@a['normal'] < np.cos(np.radians(cfg['max_normal_angle_deg'])) or
                    item['tangent']@a['tangent'] < np.cos(np.radians(cfg['max_scan_direction_angle_deg']))):
                    lane_errors.append(dict(pixel_uv=[int(u),int(v)],reason='surface_discontinuity'))
            fragment.append(item)
        if lane_errors:
            failures.append(dict(lane=lane, errors=lane_errors))
            continue
        dense_lanes.append(np.array([a['point'] for a in fragment]))
        samples = _resample(fragment, cfg['sample_step_m'])
        if lane % 2: samples.reverse()
        waypoints=[]
        for a in samples:
            point = tbc[:3,:3]@a['point']+tbc[:3,3]
            normal = _unit(tbc[:3,:3]@a['normal'])
            z = -normal; x = tbc[:3,:3]@a['tangent']; x = _unit(x-z*(x@z))
            y = _unit(np.cross(z,x)); x = _unit(np.cross(y,z))
            contact=np.eye(4);contact[:3,:3]=np.column_stack((x,y,z));contact[:3,3]=point
            stand=contact.copy();stand[:3,3]+=cfg['standoff_m']*normal
            ee_contact = contact @ np.linalg.inv(tet)
            ee_standoff = stand @ np.linalg.inv(tet)
            waypoint = dict(pixel_uv=a['pixel'].tolist(),surface_point_base_m=point.tolist(),
                normal_out_base=normal.tolist(),T_base_tcp_contact=contact.tolist(),
                T_base_ee_contact=ee_contact.tolist(),
                T_base_tcp_standoff=stand.tolist(),T_base_ee_standoff=ee_standoff.tolist())
            if tgrip is not None:
                grip_pose = ee_contact @ tgrip
                waypoint['T_base_grip_center_contact'] = grip_pose.tolist()
                waypoint['grip_center_base_m'] = grip_pose[:3, 3].tolist()
            waypoints.append(waypoint)
        points=np.array([w['surface_point_base_m'] for w in waypoints])
        segments.append(dict(lane_index=lane, width_fraction=float(fraction), waypoints=waypoints,
                             length_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                             # These are the two ends of the visible component;
                             # hidden or out-of-frame tape is outside this plan.
                             covers_visible_longitudinal_extent=True,
                             covers_both_longitudinal_ends=True))
    complete = len(segments) == lane_count
    meta=dict(executable_candidate=False, offline_preview_only=True, execution_mode='full_strip_preview',
              whole_strip_paths_complete=complete,coverage_guaranteed=False,
              eraser_footprint_checked=False,reachability_checked=False,collision_checked=False,
              between_segments_motion_planned=False,target_color=target_color,
              target_pixels=int(target.sum()),red_pixels=int(target.sum()) if target_color == 'red' else 0,
              max_segment_length_m=0.350 if target_color == 'black' else 0.200,
              white_depth_filled_points=white_filled_points if target_color == 'white' else 0,
              white_depth_reconstructed_points=white_filled_points if target_color == 'white' else 0,
              reflective_depth_reconstructed_points=white_filled_points if target_color in ('white', 'black') else 0,
              reflective_depth_fill_rms_max_mm=(max(white_fill_residuals)*1000
                                                if white_fill_residuals else None),
              strip_envelope_pixels=int(envelope.sum()),longitudinal_columns=len(columns),
              endpoint_columns=[int(round(pixel_points[0, 0])), int(round(pixel_points[-1, 0]))],requested_lanes=lane_count,
              surface_method='measured XYZ and local PCA, independent of white lettering color',
              failures=failures, config=cfg,
              gripper_center_path_included=tgrip is not None,
              visible_extent_only=True, occluded_extent_ignored=True,
              coverage_scope='currently_visible_connected_component',
              execution_note='Preview only. The visible connected component is the complete target for this scan; occluded or out-of-frame extent is ignored. Tool contact width and transitions still require validation.')
    if complete:
        dense=np.asarray(dense_lanes)
        meta['max_neighbor_lane_gap_m']=float(np.linalg.norm(np.diff(dense,axis=0),axis=2).max()) if lane_count>1 else None
        if principal_geometry:
            edges = np.stack([
                [xyz[int(round(v)), int(round(u))] for u, v in low_pixels],
                [xyz[int(round(v)), int(round(u))] for u, v in high_pixels]])
        else:
            edges=np.stack([xyz[low,columns],xyz[high,columns]])
        meta['max_strip_width_m']=float(np.linalg.norm(edges[-1]-edges[0],axis=1).max()) if np.isfinite(edges).all() else None
        # The surface path is defined by the active contact point.  The
        # gripper centre is retained separately for inspection and future
        # force-control implementations.
        meta['path_reference'] = ('configured gripper-centre wipe point'
                                  if meta.get('contact_reference') == 'wipe_center'
                                  else ('calibrated eraser front contact point'
                                        if meta.get('contact_reference') == 'eraser_front'
                                        else 'legacy calibrated eraser contact-center TCP'))
        meta['path_layout']='single_strip_centerline' if lane_count==1 else 'parallel_lanes'
        meta['surface_lane_total_length_m']=sum(s['length_m'] for s in segments)
    return dict(segments=segments,metadata=meta),target,envelope


def build(snapshot, output, rectangle=None, lane_count=1, *,
          max_surface_height_m=None, max_normal_angle_deg=None, target_color='red'):
    output, snapshot = Path(output), Path(snapshot)
    if output.exists(): raise ValueError('output already exists')
    with np.load(snapshot,allow_pickle=False) as d:
        bgr=d['bgr'].copy();xyz=d['xyz_rgb_m'].copy();meta=json.loads(str(d['metadata_json']))
    handeye=ROOT/'config/handeye_result.json';tcp=ROOT/'config/eraser_tcp_candidate.json'
    h,t,tec,tet=load_calibration(handeye,tcp)
    frames=resolve_tool_frames(t)
    if meta.get('synthetic') is not False or not meta.get('static_pose_validated'):
        raise ValueError('requires a real statically validated snapshot')
    if meta.get('camera_serial') != h['camera_serial'] or meta.get('xyz_frame') != 'C_LRGB' or meta.get('xyz_units') != 'metres':
        raise ValueError('snapshot camera/frame/units mismatch')
    tbe=_transform(meta['T_base_ee'],'capture')
    roi=make_roi(bgr.shape[:2],rectangle) if rectangle else np.ones(bgr.shape[:2],bool)
    plan,target,envelope=plan_strip(bgr,xyz,roi,tbe@tec,tet,lane_count,frames['grip'],target_color)
    plan['metadata'].update(synthetic=False,snapshot=str(snapshot.resolve()),
        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        handeye_sha256=hashlib.sha256(handeye.read_bytes()).hexdigest(),
        tcp_sha256=hashlib.sha256(tcp.read_bytes()).hexdigest(),T_base_ee_capture=tbe.tolist(),
        T_ee_tcp=tet.tolist(),F_T_EE_capture=np.array(meta['robot_before']['F_T_EE']).reshape(4,4,order='F').tolist(),
        T_ee_grip_center=frames['grip'].tolist(),
        contact_reference=frames['name'], front_contact_calibrated=frames['front_calibrated'],
        calibration_mount_unchanged_user_confirmed=False)
    plan['metadata']['path_reference'] = ('configured gripper-centre wipe point'
                                          if frames['name'] == 'wipe_center'
                                          else ('calibrated eraser front contact point'
                                                if frames['name'] == 'eraser_front'
                                                else 'legacy calibrated eraser contact-center TCP'))
    limits={}
    if max_surface_height_m is not None: limits['max_surface_height_m']=max_surface_height_m
    if max_normal_angle_deg is not None: limits['max_normal_angle_deg']=max_normal_angle_deg
    plan['metadata']['fixed_orientation_limit_overrides']=limits
    audit=[]
    for index,segment in enumerate(plan['segments']):
        points=np.array([w['surface_point_base_m'] for w in segment['waypoints']])
        normals=np.array([w['normal_out_base'] for w in segment['waypoints']])
        normal=-tbe[:3,2]
        item=dict(lane_index=segment['lane_index'],length_mm=segment['length_m']*1000,
            normal_angle_max_deg=float(np.degrees(np.arccos(np.clip(normals@normal,-1,1))).max()),
            height_variation_mm=float(np.ptp(points@normal)*1000))
        try:
            prepare_plan(plan,index,**limits);item['fixed_orientation_geometry_passed']=True
        except ExecutionError as exc:
            item.update(fixed_orientation_geometry_passed=False,reason=str(exc))
        audit.append(item)
    plan['metadata']['fixed_orientation_lane_audit']=audit
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    overlay=bgr.copy();overlay[envelope]=(.7*overlay[envelope]+.3*np.array([0,255,255])).astype(np.uint8)
    colors=[(255,0,255),(255,160,0),(0,255,0),(0,140,255),(255,255,0)]
    for seg in plan['segments']:
        uv=np.rint([w['pixel_uv'] for w in seg['waypoints']]).astype('int32')
        color=colors[seg['lane_index']%len(colors)]
        cv2.polylines(overlay,[uv],False,color,1)
        for p in [uv[0],uv[-1]]:cv2.circle(overlay,tuple(p),3,color,1)
    cv2.putText(overlay,'FULL STRIP / PREVIEW ONLY',(12,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
    cv2.imwrite(str(output/'overlay.png'),overlay)
    cv2.imwrite(str(output/'strip_envelope.png'),envelope.astype('uint8')*255)
    plot_preview(plan,xyz,envelope,tbe@tec,output/'trajectory.png')
    summary={k:v for k,v in plan['metadata'].items() if k not in ('config','T_ee_tcp','T_base_ee_capture','F_T_EE_capture')}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--roi',type=int,nargs=4);p.add_argument('--lanes',type=int,default=1)
    a=p.parse_args()
    try:result=build(a.snapshot,a.output,a.roi,a.lanes)
    except (ValueError,KeyError,ExecutionError) as exc:p.exit(1,str(exc)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if result['whole_strip_paths_complete'] else 1

if __name__=='__main__':raise SystemExit(main())
