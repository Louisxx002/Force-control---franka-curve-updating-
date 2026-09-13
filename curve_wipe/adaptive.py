"""Continuous curved TCP poses, plateau-speed passes and compensated force.

Pure computation: importing this module cannot connect to hardware.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from .execute import rigid_transform, ExecutionError
from .force import compensate_wrench


@dataclass(frozen=True)
class Settings:
    target_N: float = 1.0
    speed_m_s: float = .005
    ramp_s: float = 1.0
    period_s: float = .010
    max_normal_speed_m_s: float = .002
    max_offset_m: float = .012
    max_angular_speed_rad_s: float = np.deg2rad(5)
    max_angular_accel_rad_s2: float = np.deg2rad(15)

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in self.__dict__.values()):
            raise ValueError('settings must be positive and finite')
        if self.target_N > 1.5 or self.speed_m_s > .005:
            raise ValueError('adaptive first-version limit: <=1.5 N and <=5 mm/s')


class PassProfile:
    """Smooth acceleration, constant middle speed, smooth deceleration.

    Velocity ramps with 3u²-2u³; acceleration is zero at ramp boundaries.
    Forward/backward share exactly the same path orientation at any distance.
    """
    def __init__(self, length, speed=.005, ramp_s=1.):
        if not np.isfinite([length,speed,ramp_s]).all() or min(length,speed,ramp_s)<=0:
            raise ValueError('invalid motion profile')
        self.length=length
        self.ramp=min(ramp_s,length/speed)
        self.speed=speed
        self.cruise=max(0.,length/speed-self.ramp)
        self.duration=2*self.ramp+self.cruise

    def at(self, t, reverse=False):
        if not np.isfinite(t):raise ValueError('invalid time')
        t=np.clip(t,0,self.duration)
        if t<self.ramp:
            u=t/self.ramp
            s=self.speed*self.ramp*(u**3-.5*u**4)
            v=self.speed*(3*u*u-2*u**3)
        elif t<=self.ramp+self.cruise:
            s=.5*self.speed*self.ramp+self.speed*(t-self.ramp);v=self.speed
        else:
            u=(self.duration-t)/self.ramp
            s=self.length-self.speed*self.ramp*(u**3-.5*u**4)
            v=self.speed*(3*u*u-2*u**3)
        return (self.length-s,-v) if reverse else (s,v)


class SurfacePath:
    def __init__(self, waypoints, T_ee_tcp, reference_tcp_rotation=None):
        self.tool=rigid_transform(T_ee_tcp,'T_ee_tcp')
        self.tool_inverse=np.linalg.inv(self.tool)
        p=np.array([w['surface_point_base_m'] for w in waypoints],float)
        n=np.array([w['normal_out_base'] for w in waypoints],float)
        if p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<3 or n.shape!=p.shape or not np.isfinite([p,n]).all():
            raise ValueError('need >=3 finite positions and normals')
        if not np.allclose(np.linalg.norm(n,axis=1),1,atol=.001):raise ValueError('normals not unit')
        n=n/np.linalg.norm(n,axis=1)[:,None]
        ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
        if np.any(ds<1e-6) or np.any(ds>.010):raise ValueError('duplicate or discontinuous path')
        if np.any(np.sum(n[1:]*n[:-1],axis=1)<np.cos(np.deg2rad(25))):
            raise ValueError('normal discontinuity')
        self.arc=np.r_[0,np.cumsum(ds)];self.length=float(self.arc[-1])
        if not .005<=self.length<=.2:raise ValueError('path must be 5..200 mm')
        # Smooth normals in physical arc distance, not image pixels. Keep the
        # angular adjustment bounded so curvature is not silently flattened.
        weights=np.exp(-.5*((self.arc[:,None]-self.arc[None,:])/.012)**2)
        smooth_n=weights@n;smooth_n/=np.linalg.norm(smooth_n,axis=1)[:,None]
        adjustment=np.degrees(np.arccos(np.clip(np.sum(smooth_n*n,axis=1),-1,1)))
        self.max_normal_smoothing_deg=float(adjustment.max())
        if self.max_normal_smoothing_deg>10:
            raise ValueError('normal noise/curvature requires >10 deg smoothing; rescan or shorten segment')
        n=smooth_n
        self.position=CubicSpline(self.arc,p,axis=0,bc_type='natural')
        frames=[];last_x=None
        for i,s in enumerate(self.arc):
            # Regress tangent over the same physical neighborhood; differentiating
            # millimetre depth noise directly produces large artificial yaw.
            design=np.column_stack([self.arc-s,np.ones(len(p))])
            weight=np.sqrt(weights[i])[:,None]
            tangent=np.linalg.lstsq(design*weight,p*weight,rcond=None)[0][0]
            z=-n[i];x=tangent-z*np.dot(tangent,z)
            if np.linalg.norm(x)<1e-8:raise ValueError('degenerate tangent')
            x/=np.linalg.norm(x)
            if last_x is not None and x@last_x<0:x=-x
            y=np.cross(z,x);y/=np.linalg.norm(y);x=np.cross(y,z)
            frames.append(np.column_stack((x,y,z)));last_x=x
        if reference_tcp_rotation is not None:
            reference=np.asarray(reference_tcp_rotation,float)
            if (reference.shape!=(3,3) or not np.isfinite(reference).all() or
                    not np.allclose(reference.T@reference,np.eye(3),atol=1e-6) or
                    not np.isclose(np.linalg.det(reference),1,atol=1e-6)):
                raise ValueError('reference TCP rotation must be a proper rotation')
            # Choose the free rotation about the normal nearest the current
            # pad orientation, without changing any measured surface normal.
            local=frames[0].T@reference
            yaw=np.arctan2(local[1,0]-local[0,1],local[0,0]+local[1,1])
            twist=Rotation.from_rotvec([0,0,yaw]).as_matrix()
            frames=[frame@twist for frame in frames]
        self.orientation=RotationSpline(self.arc,Rotation.from_matrix(frames))
        dense=np.linspace(0,self.length,max(100,int(self.length/.00025)))
        smooth=self.position(dense)
        linear=np.column_stack([np.interp(dense,self.arc,p[:,k]) for k in range(3)])
        if np.max(np.linalg.norm(smooth-linear,axis=1))>.001:
            raise ValueError('position spline deviates >1 mm from measured path')
        interp_n=np.column_stack([np.interp(dense,self.arc,n[:,k]) for k in range(3)])
        interp_n/=np.linalg.norm(interp_n,axis=1)[:,None]
        fitted_n=-self.orientation(dense).as_matrix()[:,:,2]
        if np.min(np.sum(interp_n*fitted_n,axis=1))<np.cos(np.deg2rad(10)):
            raise ValueError('orientation interpolation deviates >10 deg from local normals')
        self.max_position_derivative=float(np.max(np.linalg.norm(self.position(dense,1),axis=1)))
        self.max_rotation_derivative=float(np.max(np.linalg.norm(self.orientation(dense,1),axis=1)))

    def sample(self, distance, offset=0.):
        if not np.isfinite([distance,offset]).all():raise ValueError('invalid path query')
        s=np.clip(distance,0,self.length)
        R=self.orientation(s).as_matrix();normal=-R[:,2]
        point=self.position(s)
        tcp=np.eye(4);tcp[:3,:3]=R;tcp[:3,3]=point+offset*normal
        ee=tcp@self.tool_inverse
        return point,normal,tcp,ee

    def profile(self, settings):
        # Reduce feed on sharply changing normals; never silently exceed the
        # requested speed to obtain a faster nominal pass.
        speed=min(settings.speed_m_s/max(1.,self.max_position_derivative),
                  settings.max_angular_speed_rad_s/max(self.max_rotation_derivative,1e-9))
        ramp=settings.ramp_s
        for _ in range(12):
            profile=PassProfile(self.length,speed,ramp)
            times=np.linspace(0,profile.duration,max(200,int(profile.duration/.01)))
            distance=np.array([profile.at(t)[0] for t in times])
            rotations=self.orientation(distance)
            omega=(rotations[:-1].inv()*rotations[1:]).as_rotvec()/np.diff(times)[:,None]
            accel=np.diff(omega,axis=0)/np.diff(times)[1:,None]
            if (np.max(np.linalg.norm(omega,axis=1))<=settings.max_angular_speed_rad_s*1.01 and
                    np.max(np.linalg.norm(accel,axis=1))<=settings.max_angular_accel_rad_s2):
                return profile
            speed*=.7;ramp*=1.2
        raise ValueError('cannot time-parameterize orientation within angular limits')


class CycleGuard:
    """No catch-up integration. Moderate lateness freezes progress for this cycle.

    50 ms hard stop is unchanged. This cannot make CPython hard-real-time.
    """
    def __init__(self, period=.01):self.period=period;self.late=0
    def step(self, dt, sensor_age):
        if not np.isfinite([dt,sensor_age]).all() or dt<=0 or dt>.05 or not 0<=sensor_age<=.05:
            raise ExecutionError('hard timing/stale-sensor stop')
        if dt>2*self.period:
            self.late+=1
            if self.late>=3:raise ExecutionError('repeated missed control periods')
            return 0.
        self.late=0
        return min(dt,self.period)


class ForceFeedback:
    def __init__(self, calibration, tool, settings):
        required=['T_ee_sensor','bias_sensor_6','gravity_base_N','com_sensor_m','sensor_sign']
        if calibration.get('status')!='validated_current_mount':
            raise ValueError('current-mount gravity calibration is required; fixed-pose tare is insufficient')
        from .units import WRENCH_UNITS
        if calibration.get('wrench_units') != WRENCH_UNITS:
            raise ValueError('gravity calibration must explicitly use N/Nm; legacy units are invalid')
        self.params={k:calibration[k] for k in required};self.tool=tool;self.settings=settings
        # Validate all numeric fields before any hardware connection.
        compensate_wrench(np.zeros(6),T_base_ee=np.eye(4),T_ee_tcp=tool,**self.params)
        self.filtered=None

    def read(self, raw, actual_pose, normal, dt):
        w=compensate_wrench(raw,T_base_ee=actual_pose,T_ee_tcp=self.tool,**self.params)
        fn=float(w.force_base_N@normal)
        if np.linalg.norm(w.force_base_N)>4 or np.linalg.norm(w.torque_at_tcp_base_Nm)>.5 or fn>3 or fn<-.5:
            raise ExecutionError('unfiltered compensated wrench exceeded limit')
        if self.filtered is None:self.filtered=w.force_base_N.copy()
        alpha=np.exp(-min(dt,.02)/.03)
        self.filtered=alpha*self.filtered+(1-alpha)*w.force_base_N
        return fn,float(self.filtered@normal),w,self.filtered.copy()


def load_inputs(plan_file, calibration_file, segment, settings):
    plan=json.loads(Path(plan_file).read_text())
    meta=plan['metadata']
    capture=rigid_transform(meta['T_base_ee_capture'],'capture')
    tool=rigid_transform(meta['T_ee_tcp'],'tool')
    path=SurfacePath(plan['segments'][segment]['waypoints'],tool,(capture@tool)[:3,:3])
    profile=path.profile(settings)
    calibration=json.loads(Path(calibration_file).read_text()) if calibration_file else None
    if calibration and 'F_T_EE' in calibration:
        if not np.allclose(calibration['F_T_EE'],meta['F_T_EE_capture'],atol=1e-6):
            raise ValueError('EE definition differs from gravity calibration; recalibrate')
    feedback=ForceFeedback(calibration,path.tool,settings) if calibration else None
    return plan,path,profile,feedback
