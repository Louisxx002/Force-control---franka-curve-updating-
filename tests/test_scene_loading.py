import copy
import numpy as np
import pytest
from curve_wipe.load_scene import validate_scene


def fixture():
    config = dict(group='fr3_arm', base_frame='base', flange_link='flange',
                  joint_names=['j'+str(i) for i in range(7)], namespace='/autowipe',
                  required_world_objects=['table'], required_attached_objects=['tool'])
    def box(name, frame):
        return dict(id=name, frame=frame, size_m=[.1,.2,.3], T_frame_box=np.eye(4).tolist())
    return dict(units='m', measurements_confirmed=True,
                world=[box('table','base')], attached=[box('tool','flange')]), config


def test_measured_scene():
    scene, config = fixture()
    assert validate_scene(scene, config) is scene


@pytest.mark.parametrize('case', ['unconfirmed','missing','nonfinite','frame','rotation','namespace','duplicate'])
def test_reject_scene(case):
    scene, config = fixture()
    if case == 'unconfirmed': scene['measurements_confirmed'] = False
    if case == 'missing': scene['attached'] = []
    if case == 'nonfinite': scene['world'][0]['size_m'][0] = float('nan')
    if case == 'frame': scene['world'][0]['frame'] = 'camera'
    if case == 'rotation': scene['world'][0]['T_frame_box'][0][0] = 2.
    if case == 'namespace': config['namespace'] = ''
    if case == 'duplicate': scene['world'].append(copy.deepcopy(scene['world'][0]))
    with pytest.raises(ValueError): validate_scene(scene, config)


def test_attachment_only_preserves_depth_world():
    scene, config = fixture()
    scene.pop('world')
    config['required_world_objects'] = ['realsense_observed_test']
    assert validate_scene(scene, config, attachments_only=True) is scene
    with pytest.raises(ValueError): validate_scene(scene, config)
