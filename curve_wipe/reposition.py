"""Supervised free-space alignment, followed by a straight observation move.

No force-controlled contact and no automatic recovery. Rotation pivots around
the estimated contact TCP. A fresh scan and tare are required afterwards.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .execute import rigid_transform, rotation_angle_deg


def direct_reference_return(start, target, flange, ip='172.16.0.2', speed_m_s=.015,
                            max_translation_m=.5, max_rotation_deg=45., log_path=None):
    """Guarded obstacle-free free-space return; does not read force samples."""
    start = rigid_transform(start, 'current pose'); target = rigid_transform(target, 'reference pose')
    flange = rigid_transform(flange, 'F_T_EE')
    distance = float(np.linalg.norm(target[:3, 3]-start[:3, 3]))
    angle = rotation_angle_deg(start[:3, :3], target[:3, :3])
    if distance > max_translation_m or angle > max_rotation_deg:
        raise ValueError(f'reference return exceeds {max_translation_m*1000:g} mm or {max_rotation_deg:g} degrees')
    if not np.isfinite(speed_m_s) or speed_m_s <= 0: raise ValueError('invalid return speed')
    import franky
    robot = franky.Robot(ip, relative_dynamics_factor=.05, default_torque_threshold=20., default_force_threshold=20.)
    log={'status':'starting','force_control':'disabled_in_free_space','start':start.tolist(),'target':target.tolist(),
         'distance_mm':distance*1000,'speed_mm_s':speed_m_s*1000,'samples':[]}
    try:
        state=robot.state
        if robot.has_errors or state.robot_mode != franky.RobotMode.Idle: raise RuntimeError(f'robot not ready: {state.robot_mode}')
        if not np.allclose(state.F_T_EE.matrix,flange,atol=1e-6): raise RuntimeError('F_T_EE changed')
        duration=max(1.,1.875*distance/speed_m_s);begun=time.monotonic();last=begun;previous=start.copy()
        slerp = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
        while True:
            now=time.monotonic();dt=now-last;last=now
            if dt>.05: raise RuntimeError(f'control loop stalled: dt={dt:.4f} s')
            u=min((now-begun)/duration,1.);b=u**3*(10-15*u+6*u*u)
            cmd=start.copy();cmd[:3,:3]=slerp([b]).as_matrix()[0]
            cmd[:3,3]=start[:3,3]+b*(target[:3,3]-start[:3,3])
            robot.move(franky.CartesianMotion(franky.Affine(cmd)),asynchronous=True)
            actual=np.asarray(robot.state.O_T_EE.matrix)
            if robot.has_errors or robot.state.robot_mode not in (franky.RobotMode.Idle,franky.RobotMode.Move): raise RuntimeError('robot stopped during return')
            if not np.isfinite(actual).all() or np.linalg.norm(actual[:3,3]-cmd[:3,3])>.01: raise RuntimeError('return pose tracking error')
            log['samples'].append({'t_s':now-begun,'tracking_m':float(np.linalg.norm(actual[:3,3]-cmd[:3,3]))})
            if u>=1: break
            time.sleep(max(0.,.005-(time.monotonic()-now)))
        if not robot.join_motion(5.): raise RuntimeError('return motion did not settle')
        arrived=np.asarray(robot.state.O_T_EE.matrix);error=float(np.linalg.norm(arrived[:3,3]-target[:3,3]))
        log.update(status='completed',position_error_mm=error*1000,orientation_error_deg=rotation_angle_deg(arrived[:3,:3],target[:3,:3]))
        if error>.001 or log['orientation_error_deg']>.5: raise RuntimeError('reference return error exceeds 1 mm / 0.5 degree')
        return log
    except BaseException as exc:
        log.update(status='aborted',error=str(exc)); raise
    finally:
        try: robot.stop(); robot.join_motion(3.)
        except Exception as exc: log['stop_error']=str(exc)
        if log_path: Path(log_path).write_text(json.dumps(log,indent=2)+'\n')


def build_path(spec):
    start = rigid_transform(spec['start'], 'start')
    target = rigid_transform(spec['target'], 'target')
    tool = rigid_transform(spec['T_ee_tcp'], 'tool')
    offset = tool[:3, 3]
    p0 = start[:3, 3] + start[:3, :3] @ offset
    p1 = target[:3, 3] + target[:3, :3] @ offset
    angle = rotation_angle_deg(target[:3, :3], start[:3, :3])
    max_angle=float(spec.get('max_rotation_deg',25))
    max_distance=float(spec.get('max_translation_m',.18))
    if not np.isfinite([max_angle,max_distance]).all() or not 0<max_angle<90 or max_distance<=0:
        raise ValueError('invalid return distance/rotation limits')
    if angle > max_angle or np.linalg.norm(p1-p0) > max_distance:
        raise ValueError(f'alignment exceeds {max_angle:g} degrees or {max_distance*1000:g} mm')
    normals = np.asarray(spec['normal'], float)
    points = np.asarray(spec['surface_points'], float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or
            normals.shape != (3,) or not np.isfinite(points).all() or
            not np.isfinite(normals).all() or not np.isclose(np.linalg.norm(normals), 1)):
        raise ValueError('invalid surface geometry')
    if min(np.min((p0-points)@normals), np.min((p1-points)@normals)) < .100:
        raise ValueError('TCP must stay at least 100 mm outside observed surface')
    slerp = Slerp([0,1], Rotation.from_matrix([start[:3,:3], target[:3,:3]]))
    speed_scale=float(spec.get('speed_scale',1.))
    if not np.isfinite(speed_scale) or speed_scale<=0:raise ValueError('invalid speed scale')
    rotation_time = max(2., 1.875*angle/(2.*speed_scale))
    translation_time = max(2., 1.875*np.linalg.norm(p1-p0)/(.008*speed_scale))
    return start,target,offset,p0,p1,slerp,rotation_time,translation_time


def run(spec, log_path):
    start,target,offset,p0,p1,slerp,tr,tt = build_path(spec)
    import franky
    from .serial_sensor import KunweiSensor
    robot = None
    log = {'status':'starting', 'samples':[], 'spec':spec}
    try:
        robot = franky.Robot(spec.get('robot_ip','172.16.0.2'), relative_dynamics_factor=.05,
                             default_torque_threshold=20., default_force_threshold=20.)
        s=robot.state
        if robot.has_errors or s.robot_mode != franky.RobotMode.Idle:
            raise RuntimeError(f'robot not ready: {s.robot_mode}')
        current=np.array(s.O_T_EE.matrix)
        if np.linalg.norm(current[:3,3]-start[:3,3])>.003 or rotation_angle_deg(current[:3,:3],start[:3,:3])>1:
            raise RuntimeError('robot moved after scan')
        if not np.allclose(s.F_T_EE.matrix,spec['F_T_EE'],atol=1e-6):
            raise RuntimeError('EE definition changed')
        with KunweiSensor() as sensor:
            values=[];frame=0
            for _ in range(150):
                sample=sensor.wait_next(frame);frame=sample.frame;values.append(sample.raw_sensor_6)
            bias=np.median(values,axis=0)
            if np.max(np.std(np.array(values)[:,:3],axis=0))>.1:
                raise RuntimeError('force readings not stationary')
            log['initial_wrench']=bias.tolist()
            settled_start=np.asarray(robot.state.O_T_EE.matrix)
            if np.linalg.norm(settled_start[:3,3]-start[:3,3])>.001 or rotation_angle_deg(settled_start[:3,:3],start[:3,:3])>.5:
                raise RuntimeError('robot moved during free-space baseline collection')
            previous=current.copy()
            for phase,duration in [('align_in_free_space',tr),('move_observation',tt)]:
                print(f'{phase}: {duration:.1f} s',flush=True)
                begun=time.monotonic();last=begun
                while True:
                    time.sleep(.005)
                    now=time.monotonic();dt=now-last;last=now
                    if dt>.05:raise RuntimeError('control loop stalled')
                    sample=sensor.latest(.05);delta=np.array(sample.raw_sensor_6)-bias
                    # Free-space change includes changing gravity projection.
                    # Abort on a large change; never interpret this as contact Fn.
                    if np.linalg.norm(delta[:3])>2 or np.linalg.norm(delta[3:])>.3:
                        raise RuntimeError('unexpected wrench change during free-space alignment')
                    s=robot.state;actual=np.asarray(s.O_T_EE.matrix)
                    q=np.asarray(s.q,float)
                    if not np.isfinite(actual).all() or not np.isfinite(q).all() or q[1]>=1.72:
                        raise RuntimeError('nonfinite robot state or joint-2 guard reached')
                    if robot.has_errors or s.robot_mode not in (franky.RobotMode.Idle,franky.RobotMode.Move):
                        raise RuntimeError(f'robot stopped: {s.robot_mode}')
                    if np.linalg.norm(actual[:3,3]-previous[:3,3])>.008 or rotation_angle_deg(actual[:3,:3],previous[:3,:3])>2:
                        raise RuntimeError('pose tracking error')
                    u=min((now-begun)/duration,1.);b=u**3*(10-15*u+6*u*u)
                    R=slerp(b).as_matrix() if phase=='align_in_free_space' else target[:3,:3]
                    tcp=p0 if phase=='align_in_free_space' else p0+b*(p1-p0)
                    cmd=np.eye(4);cmd[:3,:3]=R;cmd[:3,3]=tcp-R@offset
                    robot.move(franky.CartesianMotion(franky.Affine(cmd)),asynchronous=True)
                    previous=cmd
                    log['samples'].append({'phase':phase,'dt':dt,'wrench_delta':delta.tolist(),'actual':actual.tolist(),'command':cmd.tolist()})
                    if u>=1:break
                if not robot.join_motion(3.):raise RuntimeError('motion did not settle')
            arrived=np.asarray(robot.state.O_T_EE.matrix)
            error=float(np.linalg.norm(arrived[:3,3]-target[:3,3]))
            angle=rotation_angle_deg(arrived[:3,:3],target[:3,:3])
            log['arrival_position_error_mm']=error*1000
            log['arrival_orientation_error_deg']=angle
            if not np.isfinite(arrived).all() or error>.001 or angle>.5:
                raise RuntimeError('observation pose not reached within 1 mm / 0.5 degree')
            log['status']='completed'
    except BaseException as e:
        log['status']='aborted';log['error']=str(e)
        raise
    finally:
        if robot is not None:
            try:robot.stop();robot.join_motion(3.)
            except Exception as e:log['stop_error']=str(e)
        Path(log_path).write_text(json.dumps(log,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True);p.add_argument('--log',required=True)
    a=p.parse_args();run(json.loads(Path(a.spec).read_text()),a.log)
