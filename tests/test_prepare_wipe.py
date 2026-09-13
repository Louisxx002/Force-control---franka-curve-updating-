import copy,json
import numpy as np
import pytest
from curve_wipe.prepare_wipe import check_reference,prepare
from curve_wipe.reposition import build_path


def inputs():
    t=np.eye(4);t[2,3]=.3
    state=dict(O_T_EE=t.flatten(order='F').tolist(),F_T_EE=np.eye(4).flatten(order='F').tolist(),q=[0.]*7,dq=[0.]*7,robot_mode=1)
    ref=dict(name='scan_initial_pose',static_pose_validated=True,T_base_ee=t.tolist(),F_T_EE=np.eye(4).tolist(),robot_ip='test')
    return state,ref


def test_pose_tolerance_and_tool_definition():
    state,ref=inputs();assert check_reference(state,ref)['at_reference']
    state['O_T_EE'][12]=.002;assert not check_reference(state,ref)['at_reference']
    state['F_T_EE'][14]=.01
    with pytest.raises(ValueError,match='F_T_EE changed'):check_reference(state,ref)


@pytest.mark.parametrize('case',['moving','fault','joint','nan'])
def test_reject_unready_state(case):
    state,ref=inputs()
    if case=='moving':state['dq'][0]=.1
    if case=='fault':state['robot_mode']=4
    if case=='joint':state['q'][1]=1.73
    if case=='nan':state['q'][0]=float('nan')
    with pytest.raises(ValueError):check_reference(state,ref)


def test_readonly_never_captures_or_moves(tmp_path,monkeypatch):
    state,ref=inputs();state['O_T_EE'][12]=.03
    path=tmp_path/'ref.json';path.write_text(json.dumps(ref))
    monkeypatch.setattr('curve_wipe.prepare_wipe.read_state',lambda ip:state)
    def forbidden(*a,**k):raise AssertionError('unexpected hardware operation')
    monkeypatch.setattr('curve_wipe.prepare_wipe.capture',forbidden)
    monkeypatch.setattr('curve_wipe.prepare_wipe.reposition',forbidden)
    r=prepare(tmp_path/'out',path)
    assert r['status']=='return_required' and not r['motion_started']


def test_failed_return_never_runs_postreturn_scan_or_plans(tmp_path,monkeypatch):
    state,ref=inputs();state['O_T_EE'][12]=.03
    path=tmp_path/'ref.json';path.write_text(json.dumps(ref));calls=[]
    monkeypatch.setattr('curve_wipe.prepare_wipe.read_state',lambda ip:state)
    monkeypatch.setattr('curve_wipe.prepare_wipe.capture',lambda path,**kw:calls.append(path.name))
    monkeypatch.setattr('curve_wipe.prepare_wipe.return_spec',lambda *a:{})
    def fail(*a,**kw):raise RuntimeError('return failed')
    monkeypatch.setattr('curve_wipe.prepare_wipe.direct_reference_return',fail)
    with pytest.raises(RuntimeError):prepare(tmp_path/'out',path,True)
    assert calls==[]
    assert json.loads((tmp_path/'out/report.json').read_text())['status']=='blocked'


def test_return_clearance_and_size_guards():
    t=np.eye(4);t[2,3]=.3
    spec=dict(start=t.tolist(),target=t.tolist(),T_ee_tcp=np.eye(4).tolist(),normal=[0,0,1],surface_points=[[0,0,0]])
    build_path(spec)
    spec['target'][2][3]=.09
    with pytest.raises(ValueError):build_path(spec)
    spec['target'][2][3]=.6
    with pytest.raises(ValueError):build_path(spec)
