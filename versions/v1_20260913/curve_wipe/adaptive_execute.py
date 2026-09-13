"""Adaptive 1 N roundtrip entry point; preview by default, hardware is explicit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .adaptive import Settings, CycleGuard, load_inputs
from .execute import (ExecutionError, rotation_angle_deg, rigid_transform,
                      DEFAULT_MAX_APPROACH_M, validate_approach_limit, check_approach_distance)


def run(plan,path,profile,feedback,settings,log_path,ip,max_approach_m=DEFAULT_MAX_APPROACH_M):
    max_approach_m=validate_approach_limit(max_approach_m)
    if feedback is None:raise ValueError('Provide validated current-mount --calibration before execution')
    meta=plan['metadata']
    if meta.get('synthetic') is not False or meta.get('executable_candidate') is not True:
        raise ValueError('real candidate plan required')
    capture=rigid_transform(meta['T_base_ee_capture'],'capture')
    log={'status':'starting','settings':asdict(settings),'samples':[],
         'calibration':feedback.params,'plan_metadata':meta,'profile_duration_s':profile.duration,
         'max_approach_mm':max_approach_m*1000}
    log['sample_columns']=['time_s','phase','wall_dt_s','control_dt_s','path_distance_m',
        'normal_offset_m','Fn_unfiltered_N','Fn_filtered_N','raw_sensor_6',
        'compensated_force_base_N','filtered_force_base_N','nominal_tcp_base_m',
        'normal_base','command_T_base_ee','actual_T_base_ee','joint_q_rad','measured_offset_m']
    robot=None;start=time.monotonic()
    try:
        import franky
        from .serial_sensor import KunweiSensor
        robot=franky.Robot(ip,relative_dynamics_factor=.05,
                           default_torque_threshold=20.,default_force_threshold=20.)
        state=robot.state
        if robot.has_errors or state.robot_mode!=franky.RobotMode.Idle:
            raise ExecutionError(f'robot not idle: {state.robot_mode}')
        actual=np.array(state.O_T_EE.matrix)
        ready_pose=actual.copy()
        log['ready_T_base_ee']=ready_pose.tolist()
        if not np.allclose(state.F_T_EE.matrix,meta['F_T_EE_capture'],atol=1e-6):
            raise ExecutionError('EE definition changed')
        if np.linalg.norm(actual[:3,3]-capture[:3,3])>.003 or rotation_angle_deg(actual[:3,:3],capture[:3,:3])>1:
            raise ExecutionError('robot pose differs from snapshot; recapture')
        point,normal,tcp,entry=path.sample(0,.05)
        if rotation_angle_deg(actual[:3,:3],entry[:3,:3])>2:
            raise ExecutionError('align tool to preview entry orientation in free space, then rescan')
        actual_tcp=actual@path.tool
        if (actual_tcp[:3,3]-point)@normal<.04:
            raise ExecutionError('start requires free space >=40 mm')
        log['approach_distance_mm']=check_approach_distance(actual[:3,3],entry[:3,3],max_approach_m)*1000
        with KunweiSensor() as sensor:
            # Validate free-space residual; do not overwrite gravity/bias using
            # a tare at a single pose, which would invalidate other poses.
            residual=[];last_frame=0
            for _ in range(100):
                sample=sensor.wait_next(last_frame);last_frame=sample.frame
                _,_,w,_=feedback.read(sample.raw_sensor_6,actual,normal,settings.period_s)
                residual.append(w.force_base_N)
            if np.linalg.norm(np.mean(residual,axis=0))>.2 or np.max(np.std(residual,axis=0))>.1:
                raise ExecutionError('free-space compensated force is not near zero; recalibrate')
            guard=CycleGuard(settings.period_s)
            last=time.monotonic();deadline=last+settings.period_s
            command=actual.copy();distance=0.;offset=.05;phase='precontact'
            last_print=last

            def tick():
                nonlocal last,deadline,last_print
                delay=deadline-time.monotonic()
                if delay>0:time.sleep(delay)
                now=time.monotonic();wall_dt=now-last;last=now
                # Reset deadline after overrun, never issue queued catch-up ticks.
                deadline=max(deadline+settings.period_s,now+settings.period_s*.1)
                sample=sensor.latest(.05)
                control_dt=guard.step(wall_dt,sample.age_s())
                s=robot.state;actual=np.array(s.O_T_EE.matrix);q=np.array(s.q)
                if robot.has_errors or s.robot_mode not in (franky.RobotMode.Idle,franky.RobotMode.Move):
                    raise ExecutionError(f'robot fault: {s.robot_mode}')
                if not np.isfinite(q).all() or q[1]>=1.72:raise ExecutionError('joint-2 margin stop')
                if np.linalg.norm(actual[:3,3]-command[:3,3])>.01 or rotation_angle_deg(actual[:3,:3],command[:3,:3])>2:
                    raise ExecutionError('pose tracking error')
                nominal,n,_,_=path.sample(distance)
                raw_fn,filtered_fn,w,filtered=feedback.read(sample.raw_sensor_6,actual,n,wall_dt)
                measured_offset=float(((actual@path.tool)[:3,3]-nominal)@n)
                if measured_offset<-.012:raise ExecutionError('measured penetration limit')
                # Numeric rows only in the loop; conversion/plotting occur after stop.
                log['samples'].append((now-start,phase,wall_dt,control_dt,distance,offset,
                    raw_fn,filtered_fn,np.array(sample.raw_sensor_6),w.force_base_N,
                    filtered,nominal,n,command.copy(),actual.copy(),q,measured_offset))
                return control_dt,filtered_fn

            def send(target,dt):
                nonlocal command
                if dt==0:return  # No new target, no force integration, no path jump.
                if np.linalg.norm(target[:3,3]-command[:3,3])/dt>.025:
                    raise ExecutionError('combined EE command speed exceeds 25 mm/s')
                if rotation_angle_deg(target[:3,:3],command[:3,:3])/dt>np.rad2deg(settings.max_angular_speed_rad_s)*1.2:
                    raise ExecutionError('angular command speed limit')
                robot.move(franky.CartesianMotion(franky.Affine(target)),asynchronous=True)
                command=target.copy()

            def free_line(target,speed):
                from scipy.spatial.transform import Rotation,Slerp
                origin=command.copy();duration=max(1.,1.875*np.linalg.norm(target[:3,3]-origin[:3,3])/speed)
                interpolation=Slerp([0,1],Rotation.from_matrix([origin[:3,:3],target[:3,:3]]))
                elapsed=0.
                while elapsed<duration:
                    dt,f=tick()
                    if abs(f)>.3:raise ExecutionError('unexpected contact in free-space move')
                    elapsed=min(duration,elapsed+dt);u=elapsed/duration;b=u**3*(10-15*u+6*u*u)
                    # Initial orientation already matches the entry (checked above).
                    R=interpolation(b).as_matrix()
                    cmd=np.eye(4);cmd[:3,:3]=R;cmd[:3,3]=origin[:3,3]+b*(target[:3,3]-origin[:3,3])
                    send(cmd,dt)

            free_line(entry,.01)
            phase='approach';offset=.05;contact_time=0.;approach_started=time.monotonic()
            while contact_time<.10:
                dt,f=tick()
                if time.monotonic()-approach_started>65:raise ExecutionError('approach timeout')
                contact_time=contact_time+dt if f>.3 else 0.
                if f<=.3:offset-=.001*dt
                if offset<-.012:raise ExecutionError('no contact within travel limit')
                send(path.sample(0,offset)[3],dt)
            # Positive early contact is allowed up to 12 mm above vision only.
            if abs(offset)>settings.max_offset_m:raise ExecutionError('contact location disagrees with geometry')

            def force_step(f,dt):
                nonlocal offset
                velocity=np.clip(.0015*(f-settings.target_N),-settings.max_normal_speed_m_s,settings.max_normal_speed_m_s)
                offset+=velocity*dt
                if abs(offset)>settings.max_offset_m:raise ExecutionError('normal correction limit')

            phase='force_hold';stable=0.;hold_started=time.monotonic()
            while stable<1.:
                dt,f=tick();force_step(f,dt)
                stable=stable+dt if abs(f-settings.target_N)<.2 else 0.
                if time.monotonic()-hold_started>20:raise ExecutionError('force hold did not settle')
                send(path.sample(0,offset)[3],dt)
            for reverse in (False,True):
                phase='wipe_backward' if reverse else 'wipe_forward'
                progress=0.;lost=0.;begun=time.monotonic()
                while progress<profile.duration:
                    dt,f=tick();force_step(f,dt)
                    lost=lost+dt if f<.2 else 0.
                    if lost>1.:raise ExecutionError('contact lost')
                    if time.monotonic()-begun>profile.duration*3+10:raise ExecutionError('wipe timeout')
                    if f>=.2:progress=min(profile.duration,progress+dt)
                    distance,_=profile.at(progress,reverse)
                    send(path.sample(distance,offset)[3],dt)
            phase='retreat';goal=offset+.05
            while offset<goal:
                dt,f=tick();offset=min(goal,offset+.005*dt)
                send(path.sample(distance,offset)[3],dt)
            phase='return_to_precontact'
            free_line(entry,.005)
            phase='return_to_ready'
            free_line(ready_pose,.010)
            settle_started=time.monotonic()
            while True:
                dt,f=tick()
                if abs(f)>.3:raise ExecutionError('unexpected contact at ready pose')
                reached=np.asarray(robot.state.O_T_EE.matrix)
                if (np.linalg.norm(reached[:3,3]-ready_pose[:3,3])<.001 and
                        rotation_angle_deg(reached[:3,:3],ready_pose[:3,:3])<.5):break
                if time.monotonic()-settle_started>3:raise ExecutionError('return to ready did not settle')
            log['returned_to_ready']=True
            robot.join_motion(3.)
            log['status']='completed'
    except BaseException as exc:
        log['status']='aborted';log['error']=str(exc)
        raise
    finally:
        if robot is not None:
            try:robot.stop();robot.join_motion(3.)
            except Exception as e:log['cleanup_error']=str(e)
        log['duration_s']=time.monotonic()-start
        path_out=Path(log_path);path_out.parent.mkdir(parents=True,exist_ok=True)
        path_out.write_text(json.dumps(log,default=lambda a:a.tolist(),indent=2))
        summarize(log,path_out)


def summarize(log,path):
    rows=log['samples'];wipe=[r for r in rows if r[1] in ('wipe_forward','wipe_backward')]
    if not rows:return
    target=log['settings']['target_N'];report={'status':log['status'],'target_N':target}
    if wipe:
        f=np.array([r[7] for r in wipe]);h=np.array([r[5] for r in wipe])
        report.update(mean_N=float(f.mean()),std_N=float(f.std()),rmse_N=float(np.sqrt(np.mean((f-target)**2))),
                      max_N=float(f.max()),max_abs_correction_m=float(np.max(np.abs(h))))
    path.with_suffix('.summary.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,3));ax.plot([r[0] for r in rows],[r[7] for r in rows],label='Filtered Fn')
    ax.axhline(target,color='r',ls='--',label='Target');ax.set(xlabel='Time (s)',ylabel='Force (N)');ax.legend()
    fig.tight_layout();fig.savefig(path.with_suffix('.png'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--segment',type=int,default=0)
    p.add_argument('--calibration');p.add_argument('--force',type=float,default=1.)
    p.add_argument('--speed-mm-s',type=float,default=5.)
    p.add_argument('--max-approach-mm',type=float,default=DEFAULT_MAX_APPROACH_M*1000,
                   help='maximum free-space travel to precontact in mm (default: 500)')
    p.add_argument('--execute',action='store_true');p.add_argument('--ip',default='172.16.0.2')
    p.add_argument('--log',default='output/adaptive_run.json')
    a=p.parse_args();settings=Settings(target_N=a.force,speed_m_s=a.speed_mm_s/1000)
    try:
        max_approach_m=validate_approach_limit(a.max_approach_mm/1000)
    except ValueError as exc:
        p.error(str(exc))
    plan,path,profile,feedback=load_inputs(a.plan,a.calibration,a.segment,settings)
    print(json.dumps({'mode':'adaptive_curve','one_way_mm':path.length*1000,
                      'requested_mm_s':a.speed_mm_s,'effective_parameter_mm_s':profile.speed*1000,
                      'one_way_duration_s':profile.duration,'force_N':a.force,
                      'normal_smoothing_max_deg':path.max_normal_smoothing_deg,
                      'gravity_calibration_loaded':feedback is not None,
                      'max_approach_mm':max_approach_m*1000,
                      'note':'preview; actual pose, clearance, calibration and tracking checks still required'},indent=2))
    if a.execute:run(plan,path,profile,feedback,settings,a.log,a.ip,max_approach_m)


if __name__=='__main__':main()
