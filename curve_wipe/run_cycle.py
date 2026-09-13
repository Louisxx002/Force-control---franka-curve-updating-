"""Full-strip experiment: reference return, fresh planning, every lane, reference return.

Default previews only. --execute authorizes robot motion. No blind fault recovery
and no retries; only a bounded final reference-pose correction is allowed. No
automatic gripper actions. Occupancy/footprint coverage is not proven.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .plan import ROOT
from .prepare_wipe import prepare, check_reference, REFERENCE
from .capture import capture, read_state
from .full_strip import build
from .execute import prepare_plan, validate_live_pose, execute, ExecutionError
from .reposition import direct_reference_return

DEFAULT_HEIGHT_MM=30.
DEFAULT_ANGLE_DEG=45.
DEFAULT_APPROACH_MM=500.
DEFAULT_RETURN_ANGLE_DEG=120.


def validate_full_plan(plan, max_surface_height_m, max_normal_angle_deg, max_approach_m, target_color=None):
    m=plan['metadata']
    if (m.get('execution_mode')!='full_strip_preview' or m.get('synthetic') is not False or
        m.get('whole_strip_paths_complete') is not True or m.get('failures') or
        len(plan['segments'])!=m.get('requested_lanes') or not plan['segments']):
        raise ValueError('a complete real full-strip plan is required; partial coverage is rejected')
    if target_color is not None and m.get('target_color', 'red') != target_color:
        raise ValueError(f'plan target color is {m.get("target_color", "red")!r}, expected {target_color!r}')
    with np.load(m['snapshot'],allow_pickle=False) as d:scan=json.loads(str(d['metadata_json']))
    if scan.get('synthetic') is not False or not scan.get('static_pose_validated'):
        raise ValueError('snapshot must be real and stationary')
    if hashlib.sha256(Path(m['snapshot']).read_bytes()).hexdigest()!=m['snapshot_sha256']:
        raise ValueError('snapshot content changed')
    for key,file in [('handeye_sha256','handeye_result.json'),('tcp_sha256','eraser_tcp_candidate.json')]:
        if hashlib.sha256((ROOT/'config'/file).read_bytes()).hexdigest()!=m[key]:
            raise ValueError('calibration changed after planning')
    pose=np.asarray(scan['T_base_ee']);flange=np.asarray(scan['robot_before']['F_T_EE']).reshape(4,4,order='F')
    if not np.allclose(pose,m['T_base_ee_capture'],rtol=0,atol=1e-8) or not np.allclose(flange,m['F_T_EE_capture'],rtol=0,atol=1e-8):
        raise ValueError('plan and snapshot pose definitions differ')
    checks=[]
    for i,s in enumerate(plan['segments']):
        if s.get('lane_index')!=i or s.get('covers_both_longitudinal_ends') is not True:
            raise ValueError('missing or out-of-order full-length lane')
        p=prepare_plan(plan,i,max_surface_height_m=max_surface_height_m,max_normal_angle_deg=max_normal_angle_deg)
        entry=validate_live_pose(p,pose,flange,max_approach_m)
        checks.append(dict(lane=i,length_mm=p['length_m']*1000,approach_mm=float(np.linalg.norm(entry-pose[:3,3])*1000)))
    return checks


def target_consistent(original,current,tolerance_m=.008):
    if len(original['segments'])!=len(current['segments']):raise ValueError('target lane count changed')
    # Compare each lane at normalized arc length in base coordinates, not pixels.
    def samples(segment):
        points=np.asarray([w['surface_point_base_m'] for w in segment['waypoints']])
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        if arc[-1]<=0:raise ValueError('degenerate path')
        return np.column_stack([np.interp(np.linspace(0,1,51),arc/arc[-1],points[:,k]) for k in range(3)])
    error=max(float(np.linalg.norm(samples(a)-samples(b),axis=1).max()) for a,b in zip(original['segments'],current['segments']))
    if error>tolerance_m:raise ValueError(f'target shifted or reconstruction changed by {error*1000:.1f} mm; stop cycle')
    return error


def cycle(output,execute_motion=False,snapshot=None,reference_path=REFERENCE,lanes=1,
          height_mm=DEFAULT_HEIGHT_MM,angle_deg=DEFAULT_ANGLE_DEG,approach_mm=DEFAULT_APPROACH_MM,speed_scale=1.5,
          target_color='red'):
    if not np.isfinite(speed_scale) or speed_scale<=0:raise ValueError("speed scale must be finite and positive")
    if not np.isfinite([height_mm,angle_deg,approach_mm]).all() or height_mm<=0 or not 0<angle_deg<90 or approach_mm<=0:
        raise ValueError('invalid geometry/distance limits')
    if execute_motion and snapshot is not None:raise ValueError('--execute cannot use a saved snapshot; live scans required')
    output=Path(output)
    if output.exists():raise ValueError('output already exists')
    output.mkdir(parents=True)
    reference_path=Path(reference_path);reference_bytes=reference_path.read_bytes();reference=json.loads(reference_bytes)
    report=dict(status='starting',motion_requested=execute_motion,completed_lanes=[],
                max_surface_height_mm=height_mm,max_normal_angle_deg=angle_deg,max_approach_mm=approach_mm,
                contact_coverage_verified=False,automatic_retry=False,speed_scale=speed_scale,
                target_color=target_color,
                path_reference="configured contact reference",wipe_speed_mm_s=3*speed_scale)
    def save(): (output/'cycle_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def load_plan(scan,path):
        build(scan,path,lane_count=lanes,max_surface_height_m=height_mm/1000,
              max_normal_angle_deg=angle_deg,target_color=target_color)
        return json.loads((path/'plan.json').read_text())
    try:
        if execute_motion:
            print('Preparing saved initial pose and fresh scan',flush=True)
            prepare(output/'preparation',reference_path,True,max_return_m=approach_mm/1000,
                    max_return_angle_deg=DEFAULT_RETURN_ANGLE_DEG,speed_scale=speed_scale,
                    target_color=target_color)
            scan=output/'preparation/scan/snapshot.npz'
        elif snapshot is not None:scan=Path(snapshot)
        else:
            check=check_reference(read_state(reference['robot_ip']),reference)
            if not check['at_reference']:raise ValueError('preview requires current reference pose or --snapshot; no motion in preview')
            capture(output/'preview_scan',robot_ip=reference['robot_ip']);scan=output/'preview_scan/snapshot.npz'
        plan=load_plan(scan,output/'plan')
        report['path_reference'] = plan['metadata'].get('path_reference', report['path_reference'])
        options=dict(max_surface_height_m=height_mm/1000,max_normal_angle_deg=angle_deg,max_approach_m=approach_mm/1000)
        report['lane_checks']=validate_full_plan(plan,**options,target_color=target_color)
        report['plan']=str((output/'plan/plan.json').resolve());save()
        if not execute_motion:
            report['status']='preview_passed_no_motion';return report
        original=plan
        # The preparation scan is the single authoritative observation for
        # this controlled experiment. Reusing it avoids replacing a good
        # depth frame with an intermittent second capture before execution.
        report['scan_count'] = 1
        for lane in range(lanes):
            if reference_path.read_bytes()!=reference_bytes:raise ValueError('reference changed during cycle')
            if not check_reference(read_state(reference['robot_ip']),reference)['at_reference']:
                raise ValueError('not at reference before lane; no blind recovery')
            directory=output/f'lane_{lane:02d}'
            current=original
            if not check_reference(read_state(reference['robot_ip']),reference)['at_reference']:
                raise ValueError('robot left reference before lane; no blind recovery')
            report['active_lane']=lane;report['status']='executing_lane';save()
            print(f'Lane {lane+1}/{lanes}: roundtrip wipe and return',flush=True)
            log=execute(current,segment=lane,ip=reference['robot_ip'],log_path=directory/'wipe.json',
                        full_strip=True,speed_scale=speed_scale,**options)
            result=json.loads(log.read_text())
            if result['status']!='completed' or not result.get('returned_to_ready') or result.get('cleanup_errors'):
                raise RuntimeError('lane did not finish and settle cleanly')
            arrival=check_reference(read_state(reference['robot_ip']),reference)
            # The wipe executor returns to the pose captured at the start of
            # this run.  The saved scan reference can differ by a millimetre
            # after a long Cartesian round trip, so apply one small, guarded
            # free-space correction before declaring the lane complete.
            if not arrival['at_reference']:
                if arrival['position_error_mm'] > 5.0 or arrival['orientation_error_deg'] > 5.0:
                    raise RuntimeError('saved initial pose not reached after lane')
                current=read_state(reference['robot_ip'])
                correction=direct_reference_return(
                    np.asarray(current['O_T_EE']).reshape(4,4,order='F'),
                    reference['T_base_ee'],
                    np.asarray(current['F_T_EE']).reshape(4,4,order='F'),
                    ip=reference['robot_ip'], speed_m_s=.015*speed_scale,
                    max_translation_m=.005, max_rotation_deg=5.0,
                    log_path=directory/'reference_correction.json')
                report['reference_correction']=correction
                arrival=check_reference(read_state(reference['robot_ip']),reference)
            if not arrival['at_reference']:raise RuntimeError('saved initial pose not reached after lane')
            report['completed_lanes'].append(dict(lane=lane,return_check=arrival,log=str(log.resolve())))
            save()
        report.update(status='completed_all_lanes_and_returned',returned_to_initial=True)
        # Evidence image only; red tape is not assumed removable, no repeat loop.
        capture(output/'after',robot_ip=reference['robot_ip'])
        return report
    except BaseException as exc:
        report.update(status='aborted',error=str(exc),error_type=type(exc).__name__)
        raise
    finally:save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--snapshot',type=Path)
    p.add_argument('--reference',type=Path,default=REFERENCE);p.add_argument('--lanes',type=int,default=1)
    p.add_argument('--max-surface-height-mm',type=float,default=DEFAULT_HEIGHT_MM)
    p.add_argument('--max-normal-angle-deg',type=float,default=DEFAULT_ANGLE_DEG)
    p.add_argument('--speed-scale',type=float,default=1.5,help='motion speed multiplier relative to original demo')
    p.add_argument('--max-approach-mm',type=float,default=DEFAULT_APPROACH_MM)
    p.add_argument('--output',type=Path,default=ROOT/'output'/('cycle_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--target-color',choices=('red','white','black'),default='red')
    a=p.parse_args()
    try:r=cycle(a.output,a.execute,a.snapshot,a.reference,a.lanes,a.max_surface_height_mm,a.max_normal_angle_deg,a.max_approach_mm,a.speed_scale,a.target_color)
    except Exception as exc:p.exit(1,str(exc)+'\n')
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
