import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from curve_wipe.adaptive import SurfacePath,PassProfile,Settings,CycleGuard,ForceFeedback
from curve_wipe.execute import ExecutionError


def curved_path():
    x=np.linspace(0,.08,41);z=.2+1.5*x*x
    n=np.column_stack([-3*x,np.zeros(len(x)),np.ones(len(x))]);n/=np.linalg.norm(n,axis=1)[:,None]
    points=np.column_stack([x,np.zeros(len(x)),z])
    tool=np.eye(4);tool[:3,3]=[-.049,-.004,.0566]
    w=[dict(surface_point_base_m=p.tolist(),normal_out_base=a.tolist()) for p,a in zip(points,n)]
    return SurfacePath(w,tool),w


def test_contact_axis_tracks_curve_and_tcp_offset_is_rotated():
    path,_=curved_path();angles=[]
    for s in np.linspace(0,path.length,100):
        p,n,tcp,ee=path.sample(s,-.001)
        np.testing.assert_allclose((ee@path.tool)[:3,3],p-.001*n,atol=1e-12)
        np.testing.assert_allclose(tcp[:3,2],-n,atol=1e-12)
        np.testing.assert_allclose(ee@path.tool,tcp,atol=1e-12)
        angles.append(Rotation.from_matrix(tcp[:3,:3]))
    assert np.linalg.norm((angles[0].inv()*angles[-1]).as_rotvec())>np.deg2rad(10)


def test_constant_speed_middle_and_no_rotation_flip_on_return():
    path,_=curved_path();p=PassProfile(path.length,.005,1.)
    assert p.at(p.duration/2)[1]==pytest.approx(.005)
    for t in np.linspace(0,p.duration,40):
        s,_=p.at(t);back,_=p.at(p.duration-t,True)
        np.testing.assert_allclose(path.sample(s)[2],path.sample(back)[2],atol=1e-12)
    assert p.at(0)[1]==0 and p.at(p.duration)[1]==0
    assert p.duration < 1.875*path.length/.003


def test_time_parameterization_bounds_angular_speed():
    path,_=curved_path();settings=Settings();p=path.profile(settings)
    t=np.linspace(0,p.duration,2000);s=np.array([p.at(v)[0] for v in t])
    rot=path.orientation(s);v=np.linalg.norm((rot[:-1].inv()*rot[1:]).as_rotvec(),axis=1)/np.diff(t)
    assert v.max()<settings.max_angular_speed_rad_s*1.01


def test_late_tick_freezes_and_no_dt_catchup():
    g=CycleGuard(.01)
    assert g.step(.012,.002)==.01
    assert g.step(.03,.002)==0
    assert g.step(.01,.002)==.01
    with pytest.raises(ExecutionError):g.step(.0525,.002)
    with pytest.raises(ExecutionError):CycleGuard().step(.01,.051)


def test_repeated_lateness_aborts():
    g=CycleGuard()
    g.step(.025,.001);g.step(.025,.001)
    with pytest.raises(ExecutionError):g.step(.025,.001)


def test_missing_gravity_calibration_cannot_enable_adaptive_force():
    with pytest.raises(ValueError):ForceFeedback({'status':'template_not_calibrated'},np.eye(4),Settings())


def test_gravity_removed_during_rotation_and_unfiltered_trip():
    c=dict(status='validated_current_mount',wrench_units=['N','N','N','Nm','Nm','Nm'],T_ee_sensor=np.eye(4).tolist(),bias_sensor_6=[.2]*6,
           gravity_base_N=[0,0,-2],com_sensor_m=[0,0,.04],sensor_sign=1)
    f=ForceFeedback(c,np.eye(4),Settings())
    for angle in [0,.2,-.3,.5]:
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([angle,0,0]).as_matrix()
        fg=T[:3,:3].T@np.array(c['gravity_base_N']);raw=np.r_[fg,np.cross(c['com_sensor_m'],fg)]+.2
        fn,ff,w,_=f.read(raw,T,[0,0,1],.01)
        assert abs(fn)<1e-12 and abs(ff)<1e-12
    raw[:3]+=T[:3,:3].T@np.array([0,0,5])
    with pytest.raises(ExecutionError):f.read(raw,T,[0,0,1],.01)


def test_geometry_discontinuity_rejected():
    path,w=curved_path();w[10]['normal_out_base']=[0,0,-1]
    with pytest.raises(ValueError):SurfacePath(w,path.tool)


def test_reference_pad_heading_avoids_unnecessary_quarter_turn():
    path,w=curved_path()
    original=path.sample(0)[2][:3,:3]
    ref=original@Rotation.from_rotvec([0,0,np.pi/2]).as_matrix()
    aligned=SurfacePath(w,path.tool,ref)
    np.testing.assert_allclose(aligned.sample(0)[2][:3,:3],ref,atol=1e-10)
    for s in np.linspace(0,path.length,20):
        np.testing.assert_allclose(aligned.sample(s)[1],path.sample(s)[1],atol=1e-10)
        _,_,tcp,ee=aligned.sample(s)
        np.testing.assert_allclose(ee@path.tool,tcp,atol=1e-10)
