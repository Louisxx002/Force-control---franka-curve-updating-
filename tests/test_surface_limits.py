import numpy as np
import pytest
from curve_wipe.execute import prepare_plan,ExecutionError


def candidate():
    t=np.eye(4).tolist();a=np.radians(20)
    n=[np.sin(a),0,-np.cos(a)]
    return dict(metadata=dict(T_base_ee_capture=t,T_ee_tcp=t,F_T_EE_capture=t),
                segments=[dict(waypoints=[dict(surface_point_base_m=p,normal_out_base=n)
                                          for p in ([0,0,0],[.1,0,.015])])])


def test_relaxed_defaults_and_old_limits():
    p=candidate();assert prepare_plan(p)['length_m']>.1
    with pytest.raises(ExecutionError,match='15 degrees'):prepare_plan(p,max_normal_angle_deg=15)
    with pytest.raises(ExecutionError,match='10 mm'):prepare_plan(p,max_surface_height_m=.01)


@pytest.mark.parametrize('kwargs',[{'max_normal_angle_deg':90},{'max_normal_angle_deg':float('nan')},{'max_surface_height_m':0},{'max_surface_height_m':float('inf')}])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):prepare_plan(candidate(),**kwargs)
