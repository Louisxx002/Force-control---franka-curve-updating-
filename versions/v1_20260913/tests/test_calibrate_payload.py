import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from curve_wipe.calibrate_payload import fit_samples,evaluate,G,check_static


def samples(sign=1):
    rng=np.random.default_rng(15)
    mount=Rotation.from_euler('xyz',[.1,-.04,2.1]).as_matrix()
    mass=.25;com=np.array([.03,-.01,.07]);bias=np.array([.2,-.3,.1,.01,-.02,.003])
    rows=[]
    for i in range(26):
        R=Rotation.from_euler('xyz',rng.uniform(-.5,.5,3)).as_matrix()
        T=np.eye(4);T[:3,:3]=R
        g=mount.T@R.T@(mass*G)
        raw=bias+sign*np.r_[g,np.cross(com,g)]
        rows.append(dict(T_base_ee=T.tolist(),raw_mean_6=raw.tolist(),F_T_EE=np.eye(4).tolist()))
    return rows,mount,mass,com,bias


@pytest.mark.parametrize('sign',[1,-1])
def test_recover_mount_mass_bias_com_with_independent_validation(sign):
    rows,R,mass,com,bias=samples(sign)
    c=fit_samples(rows[:18],[0,0,-40])
    assert c['sensor_sign']==sign
    assert c['mass_kg']==pytest.approx(mass,abs=1e-7)
    np.testing.assert_allclose(np.array(c['T_ee_sensor'])[:3,:3],R,atol=1e-7)
    np.testing.assert_allclose(c['bias_sensor_6'],bias,atol=1e-7)
    np.testing.assert_allclose(c['com_sensor_m'],com,atol=1e-7)
    assert evaluate(c,rows[18:])['force_max_N']<1e-7


def test_repeated_poses_cannot_fit_mount():
    rows,*_=samples()
    with pytest.raises(ValueError,match='激励不足'):fit_samples([rows[0]]*16,[0,0,0])


def test_bad_holdout_reveals_contact_or_wrong_model():
    rows,*_=samples();c=fit_samples(rows[:18],[0,0,0])
    rows[-1]['raw_mean_6'][0]+=1
    assert evaluate(c,rows[18:])['force_max_N']>.9


def test_stationarity_rejects_movement():
    s=dict(O_T_EE=np.eye(4).flatten(order='F').tolist(),F_T_EE=np.eye(4).flatten(order='F').tolist(),dq=[0]*7)
    changed=dict(s);T=np.eye(4);T[0,3]=.002;changed['O_T_EE']=T.flatten(order='F').tolist()
    with pytest.raises(ValueError,match='移动'):check_static([s,changed])
