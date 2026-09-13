"""Return to the saved observation pose, then freshly scan and plan the full strip.

Default is a read-only pose check. --execute authorizes guarded free-space return
and camera capture, but never starts contact wiping or changes the gripper.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from .capture import read_state, capture
from .execute import rigid_transform, rotation_angle_deg
from .plan import ROOT, load_calibration
from .full_strip import strip_envelope, build as build_strip
from .reposition import build_path, run as reposition, direct_reference_return
from .geometry import TARGET_COLORS

REFERENCE = ROOT/'config/scan_initial_pose.json'


def check_reference(state, reference):
    if reference.get('name') != 'scan_initial_pose' or reference.get('static_pose_validated') is not True:
        raise ValueError('reference must be a statically validated scan_initial_pose')
    actual=rigid_transform(np.asarray(state['O_T_EE']).reshape(4,4,order='F'),'current pose')
    target=rigid_transform(reference['T_base_ee'],'reference pose')
    flange=rigid_transform(np.asarray(state['F_T_EE']).reshape(4,4,order='F'),'current F_T_EE')
    expected=rigid_transform(reference['F_T_EE'],'reference F_T_EE')
    if not np.allclose(flange,expected,rtol=0,atol=1e-6):raise ValueError('F_T_EE changed since reference recording')
    q,dq=np.asarray(state['q'],float),np.asarray(state['dq'],float)
    if q.shape!=(7,) or dq.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite(dq).all():
        raise ValueError('invalid joint state')
    if state['robot_mode'] != 1 or np.max(np.abs(dq))>=.01:raise ValueError('robot must be idle and stationary')
    if q[1]>=1.72:raise ValueError('joint 2 exceeds existing guard')
    distance=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
    angle=rotation_angle_deg(actual[:3,:3],target[:3,:3])
    return dict(at_reference=distance<=.001 and angle<=.5,position_error_mm=distance*1000,
                orientation_error_deg=angle)


def return_spec(snapshot, reference, max_return_m=.18, max_return_angle_deg=25., target_color='red'):
    with np.load(snapshot,allow_pickle=False) as d:
        metadata=json.loads(str(d['metadata_json']));bgr=d['bgr'];xyz=d['xyz_rgb_m']
    if metadata.get('synthetic') is not False or not metadata.get('static_pose_validated'):
        raise ValueError('return requires a fresh real stationary scan')
    check_reference(metadata['robot_before'],reference)
    h,t,tec,tet=load_calibration(ROOT/'config/handeye_result.json',ROOT/'config/eraser_tcp_candidate.json')
    if metadata.get('camera_serial')!=h['camera_serial'] or metadata.get('xyz_frame')!='C_LRGB' or metadata.get('xyz_units')!='metres':
        raise ValueError('scan camera/frame/units mismatch')
    _,mask,_,_,_=strip_envelope(bgr,np.ones(bgr.shape[:2],bool),target_color)
    if not (np.isfinite(xyz[mask]).all() and (xyz[mask,2]>0).all()):
        raise ValueError('missing strip depth; cannot verify return clearance')
    start=rigid_transform(metadata['T_base_ee'],'scan pose')
    tbc=start@tec
    points=xyz[mask]@tbc[:3,:3].T+tbc[:3,3]
    spec=dict(start=start.tolist(),target=reference['T_base_ee'],T_ee_tcp=tet.tolist(),
              F_T_EE=reference['F_T_EE'],normal=(-start[:3,2]).tolist(),
              surface_points=points.tolist(),robot_ip=reference['robot_ip'],
              max_translation_m=max_return_m,max_rotation_deg=max_return_angle_deg)
    build_path(spec)
    return spec


def prepare(output, reference_path=REFERENCE, execute=False, max_return_m=.18, max_return_angle_deg=25., speed_scale=1., target_color='red'):
    output,reference_path=Path(output),Path(reference_path)
    if output.exists():raise ValueError('output already exists')
    reference_bytes=reference_path.read_bytes();reference=json.loads(reference_bytes)
    output.mkdir(parents=True)
    report=dict(status='checking_reference',motion_requested=execute,motion_started=False,
                contact_wiping_started=False,target_color=target_color,
                reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                reference=str(reference_path.resolve()))
    try:
        state=read_state(reference['robot_ip'])
        report['initial_check']=check_reference(state,reference)
        if not execute:
            report['status']='at_reference' if report['initial_check']['at_reference'] else 'return_required'
            report['note']='read-only check; no scan or motion. Use --execute for return then fresh full-strip planning.'
            return report
        if not report['initial_check']['at_reference']:
            # The user has declared the cell obstacle-free. If the target is
            # not visible from the abandoned pose, return in air using only
            # the saved reference; the camera scan is performed afterwards.
            current=read_state(reference['robot_ip'])
            direct_log=direct_reference_return(np.asarray(current['O_T_EE']).reshape(4,4,order='F'), reference['T_base_ee'],
                np.asarray(current['F_T_EE']).reshape(4,4,order='F'), ip=reference['robot_ip'], speed_m_s=.015*speed_scale,
                max_translation_m=max_return_m, max_rotation_deg=max_return_angle_deg,
                log_path=output/'return_log.json')
            report['direct_air_return']=direct_log
            # Do not move toward an obsolete reference if the file was changed.
            if reference_path.read_bytes()!=reference_bytes:raise ValueError('reference changed during preparation')
            report['motion_started']=True
        state=read_state(reference['robot_ip']);report['arrival_check']=check_reference(state,reference)
        if not report['arrival_check']['at_reference']:raise ValueError('reference pose not reached; no post-return plan')
        capture(output/'scan',robot_ip=reference['robot_ip'])
        with np.load(output/'scan/snapshot.npz',allow_pickle=False) as d:meta=json.loads(str(d['metadata_json']))
        if not meta.get('static_pose_validated'):raise ValueError('post-return scan not stationary')
        for key in ('robot_before','robot_after'):
            if not check_reference(meta[key],reference)['at_reference']:raise ValueError('robot left reference during scan')
        plan=build_strip(output/'scan/snapshot.npz',output/'full_strip',target_color=target_color)
        report['full_strip_plan']=str((output/'full_strip/plan.json').resolve())
        report['status']='full_strip_preview_ready' if plan['whole_strip_paths_complete'] else 'full_strip_geometry_blocked'
        report['executable']=False
        report['note']='Preparation only; full-strip contact execution and lane transitions are not integrated.'
        return report
    except Exception as exc:
        report.update(status='blocked',error=str(exc),error_type=type(exc).__name__)
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,default=REFERENCE)
    p.add_argument('--output',type=Path,default=ROOT/'output'/('prepare_wipe_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
    p.add_argument('--execute',action='store_true',help='authorize guarded return and fresh scan; no contact wiping')
    p.add_argument('--max-return-mm',type=float,default=180.)
    p.add_argument('--max-return-angle-deg',type=float,default=25.)
    p.add_argument('--speed-scale',type=float,default=1.)
    p.add_argument('--target-color',choices=TARGET_COLORS,default='red',
                   help='target tape color; detection uses HSV plus elongated-component selection')
    a=p.parse_args()
    try:r=prepare(a.output,a.reference,a.execute,a.max_return_mm/1000,a.max_return_angle_deg,a.speed_scale,a.target_color)
    except Exception as exc:p.exit(1,str(exc)+'\n')
    print(json.dumps(r,indent=2))
    return 1 if r['status']=='full_strip_geometry_blocked' else 0

if __name__=='__main__':raise SystemExit(main())
