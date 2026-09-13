import copy,json
from pathlib import Path
import numpy as np
import pytest
from curve_wipe.run_cycle import cycle,target_consistent,validate_full_plan


def test_partial_plan_never_admitted():
    p={'metadata':dict(execution_mode='full_strip_preview',synthetic=False,whole_strip_paths_complete=False),'segments':[]}
    with pytest.raises(ValueError,match='complete real'):validate_full_plan(p,.03,45,.5)


def test_shifted_target_rejected():
    def plan(y):return {'segments':[{'waypoints':[{'surface_point_base_m':[x,y,0]} for x in [0,.05,.1]]}]}
    assert target_consistent(plan(0),plan(.001))==pytest.approx(.001)
    with pytest.raises(ValueError,match='shifted'):target_consistent(plan(0),plan(.01))


def test_historical_snapshot_cannot_execute(tmp_path):
    with pytest.raises(ValueError,match='live scans required'):
        cycle(tmp_path/'out',True,snapshot='old.npz')
    assert not (tmp_path/'out').exists()


def test_lane_failure_stops_cycle_and_does_not_retry(tmp_path,monkeypatch):
    ref=tmp_path/'ref.json';ref.write_text(json.dumps({'robot_ip':'test'}))
    events=[]
    fake_plan={'metadata':{},'segments':[{},{}]}
    def fake_capture(path,**kw):
        events.append(('capture',str(path)));Path(path).mkdir(parents=True)
        np.savez(Path(path)/'snapshot.npz',metadata_json=json.dumps({'robot_before':{},'robot_after':{}}))
    def fake_build(scan,path,**kw):
        Path(path).mkdir(parents=True);(Path(path)/'plan.json').write_text(json.dumps(fake_plan))
    def fake_execute(*a,**kw):events.append(('execute',kw['segment']));raise RuntimeError('force protection')
    monkeypatch.setattr('curve_wipe.run_cycle.prepare',lambda *a,**kw:None)
    monkeypatch.setattr('curve_wipe.run_cycle.read_state',lambda ip:{})
    monkeypatch.setattr('curve_wipe.run_cycle.check_reference',lambda *a:{'at_reference':True})
    monkeypatch.setattr('curve_wipe.run_cycle.capture',fake_capture)
    monkeypatch.setattr('curve_wipe.run_cycle.build',fake_build)
    monkeypatch.setattr('curve_wipe.run_cycle.validate_full_plan',lambda *a,**kw:[])
    monkeypatch.setattr('curve_wipe.run_cycle.target_consistent',lambda *a:0)
    monkeypatch.setattr('curve_wipe.run_cycle.execute',fake_execute)
    with pytest.raises(RuntimeError,match='force protection'):cycle(tmp_path/'out',True,reference_path=ref,lanes=2)
    assert [v for k,v in events if k=='execute']==[0]
    r=json.loads((tmp_path/'out/cycle_report.json').read_text())
    assert r['status']=='aborted' and not r['completed_lanes']


def test_return_speed_scale_shortens_motion_duration():
    from curve_wipe.reposition import build_path
    from scipy.spatial.transform import Rotation
    t=np.eye(4);t[2,3]=.3;end=t.copy();end[0,3]=.12;end[:3,:3]=Rotation.from_euler('z',15,degrees=True).as_matrix()
    spec=dict(start=t.tolist(),target=end.tolist(),T_ee_tcp=np.eye(4).tolist(),normal=[0,0,1],surface_points=[[0,0,0]])
    before=build_path(spec);spec['speed_scale']=1.5;after=build_path(spec)
    assert after[-1]==pytest.approx(before[-1]/1.5)
    assert after[-2]==pytest.approx(before[-2]/1.5)
