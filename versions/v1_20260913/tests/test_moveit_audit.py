"""Service-contract tests with mocked responses, not physical scene validation."""
import json
from unittest.mock import MagicMock
import numpy as np
import pytest

pytest.importorskip('rclpy')
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from moveit_msgs.srv import GetPlanningScene, GetPositionFK, GetCartesianPath, GetStateValidity
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint
from curve_wipe.check_approach import audit


@pytest.mark.parametrize('case', ['success', 'missing_geometry', 'partial', 'fk_mismatch', 'collision'])
def test_service_contract(tmp_path, monkeypatch, case):
    names = ['j'+str(i) for i in range(7)]
    config = dict(group='arm', base_frame='base', flange_link='flange', joint_names=names,
                  required_world_objects=['table'], required_attached_objects=['tool'])
    start = np.eye(4)
    end = start.copy(); end[0, 3] = .05
    snapshot = tmp_path/'snapshot.npz'
    np.savez(snapshot, metadata_json=json.dumps(dict(synthetic=False, static_pose_validated=True,
        T_base_ee=start.tolist(), robot_before=dict(q=[0.]*7, F_T_EE=start.flatten(order='F').tolist()))))
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps(dict(metadata=dict(snapshot=str(snapshot),
        T_base_ee_capture=start.tolist(), F_T_EE_capture=start.tolist()))))
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(snapshot=str(snapshot), selected_candidate=0,
        candidates=[dict(plan=str(plan), approach=dict(poses_base_ee=[start.tolist(), end.tolist()]))])))
    scene = GetPlanningScene.Response()
    solid = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[.1, .1, .1])
    object_pose = Pose(); object_pose.orientation.w = 1.
    table = CollisionObject(id='table', primitives=[solid], primitive_poses=[object_pose])
    if case != 'missing_geometry': scene.scene.world.collision_objects = [table]
    tool = AttachedCollisionObject(link_name='flange', object=CollisionObject(
        id='tool', primitives=[solid], primitive_poses=[object_pose]))
    scene.scene.robot_state.attached_collision_objects = [tool]
    fk_calls = 0
    queried = []

    def create_client(kind, name):
        client = MagicMock()
        client.wait_for_service.return_value = True
        def call_async(request):
            nonlocal fk_calls
            queried.append(name)
            if kind is GetPlanningScene:
                response = scene
            elif kind is GetPositionFK:
                response = GetPositionFK.Response()
                response.error_code.val = 1
                response.fk_link_names = ['flange']
                pose = PoseStamped(); pose.header.frame_id = 'base'; pose.pose.orientation.w = 1.
                pose.pose.position.x = (0. if fk_calls == 0 else .05)
                if case == 'fk_mismatch': pose.pose.position.z = .01
                response.pose_stamped = [pose]; fk_calls += 1
            elif kind is GetCartesianPath:
                assert request.avoid_collisions
                assert request.max_step == .005
                assert request.link_name == 'flange'
                response = GetCartesianPath.Response(); response.error_code.val = 1
                response.fraction = .5 if case == 'partial' else 1.
                response.solution.joint_trajectory.joint_names = names
                for i in range(2):
                    point = JointTrajectoryPoint(positions=[float(i)*.02]+[0.]*6)
                    point.time_from_start.sec = i
                    response.solution.joint_trajectory.points.append(point)
            elif kind is GetStateValidity:
                response = GetStateValidity.Response(valid=case != 'collision')
            else:
                raise AssertionError('unexpected service: '+name)
            future = MagicMock(); future.done.return_value = True; future.result.return_value = response
            return future
        client.call_async.side_effect = call_async
        return client
    node = MagicMock(); node.create_client.side_effect = create_client
    monkeypatch.setattr('rclpy.node.Node', lambda name: node)
    monkeypatch.setattr('rclpy.init', lambda: None)
    monkeypatch.setattr('rclpy.shutdown', lambda: None)
    monkeypatch.setattr('rclpy.spin_until_future_complete', lambda *a, **k: None)
    if case == 'success':
        result = audit(report, config)
        assert result['status'] == 'sampled_scene_audit_passed'
        assert result['motion_started'] is False and result['executable'] is False
        assert result['collision_audit_samples'] == 3
    else:
        with pytest.raises(ValueError): audit(report, config)
    assert all(name in ('/get_planning_scene', '/compute_fk', '/compute_cartesian_path',
                        '/check_state_validity') for name in queried)
    node.destroy_node.assert_called_once()
