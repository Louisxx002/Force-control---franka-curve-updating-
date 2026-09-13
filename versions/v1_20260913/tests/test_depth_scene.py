import json
import numpy as np
import pytest
from curve_wipe.depth_scene import voxelize, build


def test_voxels_cover_points_after_transform():
    xyz = np.array([[[.001,.002,.103],[.009,.005,.109]],
                    [[-.003,.007,.107],[float('nan'),0,0]]])
    transform = np.eye(4); transform[:3,3] = [.5,-.2,.1]
    centers, stats = voxelize(xyz, transform, .02, .01)
    points = xyz[np.isfinite(xyz).all(axis=-1)] + transform[:3,3]
    assert all(np.any(np.max(np.abs(centers-p),axis=1) <= stats['box_size_m']/2) for p in points)
    assert stats['valid_points'] == 3 and stats['invalid_pixels'] == 1


def test_holes_do_not_create_geometry():
    xyz = np.array([[[0.,0.,1.],[np.nan,np.nan,np.nan],[1.,0.,1.]]])
    centers, stats = voxelize(xyz,np.eye(4))
    assert len(centers) == 2
    assert not np.any((centers[:,0] > .1) & (centers[:,0] < .9))


def test_limit_rejects_instead_of_dropping():
    with pytest.raises(ValueError, match='silently dropped'):
        voxelize(np.array([[[0.,0.,1.],[1.,0.,1.]]]),np.eye(4),max_voxels=1)


@pytest.mark.parametrize('voxel,padding', [(0,.01),(.1,.01),(.02,-.01),(.02,float('nan'))])
def test_bad_resolution(voxel,padding):
    with pytest.raises(ValueError): voxelize(np.ones((1,1,3)),np.eye(4),voxel,padding)


def test_build_binds_snapshot_and_calibration(tmp_path):
    handeye=tmp_path/'handeye.json'
    handeye.write_text(json.dumps(dict(camera_serial='test',T_EE_L_C_LRGB=np.eye(4).tolist(),T_C_LRGB_EE_L=np.eye(4).tolist())))
    snapshot=tmp_path/'scan.npz'
    metadata=dict(synthetic=False,static_pose_validated=True,xyz_frame='C_LRGB',xyz_units='metres',camera_serial='test',T_base_ee=np.eye(4).tolist())
    np.savez(snapshot,xyz_rgb_m=np.ones((2,2,3)),metadata_json=json.dumps(metadata))
    result,_=build(snapshot,handeye,tmp_path/'out')
    config=json.loads((tmp_path/'out/motion_config.json').read_text())
    assert result['executable'] is False
    assert config['required_world_objects'] == [result['object_id']]
    assert config['required_attached_objects']
    assert len(result['snapshot_sha256']) == 64
    metadata['synthetic']=True
    np.savez(snapshot,xyz_rgb_m=np.ones((2,2,3)),metadata_json=json.dumps(metadata))
    with pytest.raises(ValueError,match='real snapshot'): build(snapshot,handeye,tmp_path/'bad')
