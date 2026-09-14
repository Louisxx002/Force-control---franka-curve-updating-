import numpy as np
import pytest
from curve_wipe.full_strip import plan_strip,strip_envelope
from curve_wipe.execute import prepare_plan,ExecutionError,normal_at,rotation_at


def scene():
    v,u=np.indices((61,201));xyz=np.stack([(u-100)*.001,(v-30)*.001,np.full(u.shape,.5)],axis=-1)
    bgr=np.zeros((61,201,3),np.uint8);bgr[20:41,20:181,2]=255
    # Lettering stays inside one connected strip but breaks individual red rows.
    bgr[22:39,70:90]=255;bgr[23:38,120:135]=255
    return bgr,xyz,np.ones((61,201),bool)


def test_letters_do_not_split_full_length_lanes():
    bgr,xyz,roi=scene()
    plan,red,envelope=plan_strip(bgr,xyz,roi,np.eye(4),np.eye(4),lane_count=5)
    assert envelope.sum() > red.sum()
    assert len(plan['segments']) == 5 and plan['metadata']['whole_strip_paths_complete']
    assert plan['metadata']['visible_extent_only'] is True
    assert plan['metadata']['coverage_scope'] == 'currently_visible_connected_component'
    for s in plan['segments']:
        us=[w['pixel_uv'][0] for w in s['waypoints']]
        assert min(us) == 20 and max(us) == 180
        assert s['length_m'] == pytest.approx(.16)
    assert plan['metadata']['coverage_guaranteed'] is False
    assert plan['metadata']['executable_candidate'] is False
    with pytest.raises(ExecutionError,match='execution requires'):
        prepare_plan(plan,require_real=True)


def test_white_target_mode_selects_elongated_bright_component():
    bgr=np.full((61,201,3),145,np.uint8)
    bgr[20:31,20:181]=220
    xyz=np.stack([np.indices((61,201))[1]*.001,
                  np.indices((61,201))[0]*.001,
                  np.full((61,201),.5)],axis=-1)
    plan,target,envelope=plan_strip(bgr,xyz,np.ones((61,201),bool),
                                    np.eye(4),np.eye(4),target_color='white')
    assert target.sum() > 1000 and envelope.sum() >= target.sum()
    assert plan['metadata']['target_color']=='white'
    assert plan['metadata']['whole_strip_paths_complete']


def test_diagonal_white_target_uses_principal_axis():
    import cv2
    bgr=np.full((100,240,3),145,np.uint8)
    polygon=np.array([[25,25],[205,75],[201,88],[21,38]],np.int32)
    cv2.fillPoly(bgr,[polygon],(235,235,235))
    yy,xx=np.indices((100,240))
    xyz=np.stack([xx*.001,yy*.001,np.full(xx.shape,.5)],axis=-1)
    plan,target,envelope=plan_strip(bgr,xyz,np.ones((100,240),bool),
                                    np.eye(4),np.eye(4),target_color='white')
    assert target.sum() > 1000 and envelope.sum() >= target.sum()
    assert plan['metadata']['whole_strip_paths_complete']
    assert plan['metadata']['target_color']=='white'


def test_depth_hole_blocks_whole_lanes_instead_of_short_fallback():
    bgr,xyz,roi=scene();xyz[:,98:103]=np.nan
    plan,_,_=plan_strip(bgr,xyz,roi,np.eye(4),np.eye(4),lane_count=5)
    assert not plan['metadata']['whole_strip_paths_complete']
    assert not plan['segments']
    assert len(plan['metadata']['failures']) == 5


def test_depth_discontinuity_does_not_connect():
    bgr,xyz,roi=scene();xyz[:,100:,2]+=.1
    plan,_,_=plan_strip(bgr,xyz,roi,np.eye(4),np.eye(4),lane_count=5)
    assert not plan['metadata']['whole_strip_paths_complete']
    assert not plan['segments']


def test_two_separate_targets_require_roi():
    bgr,xyz,roi=scene();bgr[46:55,40:150,2]=255
    with pytest.raises(ValueError,match='exactly one'):strip_envelope(bgr,roi)


def test_clipped_roi_defines_visible_extent():
    bgr,xyz,roi=scene();roi[:,:50]=False
    target, envelope, columns, low, high = strip_envelope(bgr,roi)
    assert columns[0] == 50 and columns[-1] == 180
    assert target.sum() > 0 and envelope.sum() >= target.sum()


def test_tcp_transform_and_lane_spacing():
    bgr,xyz,roi=scene();tbc=np.eye(4);tbc[:3,3]=[.2,.3,.4];tet=np.eye(4);tet[2,3]=.15
    plan,_,_=plan_strip(bgr,xyz,roi,tbc,tet,lane_count=5)
    assert plan['metadata']['max_neighbor_lane_gap_m'] == pytest.approx(.005)
    for s in plan['segments']:
        for w in s['waypoints']:
            np.testing.assert_allclose(np.array(w['T_base_ee_contact'])@tet,w['T_base_tcp_contact'],atol=1e-12)
            np.testing.assert_allclose(np.array(w['T_base_tcp_contact'])[:3,:3].T@np.array(w['T_base_tcp_contact'])[:3,:3],np.eye(3),atol=1e-12)


def test_default_is_single_centerline_using_contact_tcp():
    from curve_wipe.execute import ee_position
    bgr,xyz,roi=scene();tool=np.eye(4);tool[:3,3]=[.04,0,.15]
    plan,_,_=plan_strip(bgr,xyz,roi,np.eye(4),tool)
    assert len(plan['segments'])==1
    segment=plan['segments'][0]
    assert segment['width_fraction']==.5
    np.testing.assert_allclose([w['pixel_uv'][1] for w in segment['waypoints']],30,atol=1e-12)
    assert segment['waypoints'][0]['pixel_uv'][0]==20
    assert segment['waypoints'][-1]['pixel_uv'][0]==180
    for w in segment['waypoints']:
        point=np.array(w['surface_point_base_m'])
        ee=ee_position(point,0,np.eye(3),tool[:3,3],np.array([0,0,-1]))
        np.testing.assert_allclose(ee+tool[:3,3],point,atol=1e-12)


def test_gripper_center_is_exported_separately_from_contact_point():
    bgr,xyz,roi=scene(); tool=np.eye(4); tool[2,3]=.15
    grip=np.eye(4); grip[2,3]=.10
    plan,_,_=plan_strip(bgr,xyz,roi,np.eye(4),tool,lane_count=1,tgrip=grip)
    w=plan['segments'][0]['waypoints'][0]
    ee=np.array(w['T_base_ee_contact'])
    expected=ee@grip
    np.testing.assert_allclose(np.array(w['T_base_grip_center_contact']),expected,atol=1e-12)
    np.testing.assert_allclose(np.array(w['grip_center_base_m']),expected[:3,3],atol=1e-12)
    assert plan['metadata']['gripper_center_path_included'] is True


def test_spatial_rotation_keeps_tcp_z_on_interpolated_surface_normal():
    normals = np.array([[0., 0., 1.], [.24, .08, np.sqrt(1-.24**2-.08**2)],
                        [-.15, .12, np.sqrt(1-.15**2-.12**2)]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    tcp_frames = []
    for n in normals:
        z = -n
        x = np.array([1., 0., 0.])
        x -= z * (x @ z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        y /= np.linalg.norm(y)
        tcp_frames.append(np.column_stack((x, y, z)))
    prepared = {
        'spatial_orientation': True,
        'arc': np.array([0., .1, .2]),
        'normals': normals,
        'tcp_frames': np.asarray(tcp_frames),
        'tool_rotation': np.array([[np.cos(.3), -np.sin(.3), 0.],
                                   [np.sin(.3),  np.cos(.3), 0.],
                                   [0., 0., 1.]]),
    }
    for distance in np.linspace(0., .2, 9):
        ee_rotation = rotation_at(prepared, distance)
        tcp_z = ee_rotation @ prepared['tool_rotation'][:, 2]
        np.testing.assert_allclose(tcp_z, -normal_at(prepared, distance), atol=1e-12)
        np.testing.assert_allclose(ee_rotation.T @ ee_rotation, np.eye(3), atol=1e-12)
