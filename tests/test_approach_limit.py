import numpy as np
import pytest
from curve_wipe.execute import (ExecutionError, check_approach_distance,
                                validate_approach_limit, validate_live_pose)


def test_controlled_experiment_distance_and_explicit_old_limit():
    prepared=dict(rotation=np.eye(3),flange=np.eye(4),tcp_offset=np.zeros(3),
                  points=np.array([[0.,0.,0.],[.01,0.,0.]]),normal=np.array([0.,0.,-1.]))
    pose=np.eye(4);pose[2,3]=-.35
    entry=validate_live_pose(prepared,pose,np.eye(4))
    assert np.allclose(entry,[0,0,-.05])
    with pytest.raises(ExecutionError,match='limit is 120'):
        validate_live_pose(prepared,pose,np.eye(4),.12)
    pose[2,3]=-.56
    with pytest.raises(ExecutionError,match='limit is 500'):
        validate_live_pose(prepared,pose,np.eye(4))
    pose[2,3]=-.01
    with pytest.raises(ExecutionError,match='20 mm'):
        validate_live_pose(prepared,pose,np.eye(4),.8)


def test_three_dimensional_distance_and_custom_limit():
    with pytest.raises(ExecutionError): check_approach_distance([0,0,0],[.4,.4,0])
    assert check_approach_distance([0,0,0],[.4,.4,0],.6) == pytest.approx(np.sqrt(.32))
    assert check_approach_distance([0,0,0],[0,0,.5]) == .5


@pytest.mark.parametrize('value',[0,-1,float('inf'),float('nan')])
def test_bad_limits(value):
    with pytest.raises(ValueError): validate_approach_limit(value)
