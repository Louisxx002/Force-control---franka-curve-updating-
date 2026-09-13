"""Read-only MoveIt service audit. No controller, execution action or scene mutation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from .approach_checks import (validate_config, flange_waypoints, check_fk,
                              check_trajectory, dense_joint_samples)


def pose_matrix(pose):
    q = pose.orientation
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def audit(report_path, config):
    validate_config(config)
    report = json.loads(Path(report_path).read_text())
    candidate = report['candidates'][report['selected_candidate']]
    plan = json.loads(Path(candidate['plan']).read_text())
    depth_scene = config.get('observed_depth_scene')
    if depth_scene is not None:
        if depth_scene['snapshot_sha256'] != hashlib.sha256(Path(report['snapshot']).read_bytes()).hexdigest():
            raise ValueError('depth scene and trajectory use different snapshots; rebuild scene')
        if depth_scene['handeye_sha256'] != plan['metadata'].get('handeye_sha256'):
            raise ValueError('depth scene and trajectory use different handeye calibration')
        if depth_scene['object_id'] not in config['required_world_objects']:
            raise ValueError('depth scene object must be required for audit')
    with np.load(report['snapshot'], allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
    if metadata.get('synthetic') is not False or not metadata.get('static_pose_validated'):
        raise ValueError('audit requires a real statically validated snapshot')
    # Bind the plan to the same capture used for q and flange definition.
    if Path(plan['metadata']['snapshot']).resolve() != Path(report['snapshot']).resolve():
        raise ValueError('plan and report refer to different snapshots')
    capture = np.array(metadata['T_base_ee'])
    if not np.allclose(capture, plan['metadata']['T_base_ee_capture'], atol=1e-8, rtol=0):
        raise ValueError('plan capture pose mismatch')
    flange = np.array(metadata['robot_before']['F_T_EE']).reshape(4, 4, order='F')
    if not np.allclose(flange, plan['metadata']['F_T_EE_capture'], atol=1e-8, rtol=0):
        raise ValueError('plan flange definition mismatch')
    poses = flange_waypoints(candidate['approach']['poses_base_ee'], flange)
    check_fk(capture @ np.linalg.inv(flange), poses[0])
    start_q = metadata['robot_before']['q']
    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.srv import GetPlanningScene, GetPositionFK, GetCartesianPath, GetStateValidity
    from rosidl_runtime_py.convert import message_to_ordereddict
    from rclpy.node import Node
    rclpy.init()
    node = Node('autowipe_readonly_audit')
    prefix = config.get('namespace', '').rstrip('/')
    started = time.monotonic()

    def call(kind, name, request):
        if time.monotonic()-started > 45:
            raise RuntimeError('planning audit exceeded 45 s; no motion requested')
        client = node.create_client(kind, prefix+'/'+name)
        try:
            if not client.wait_for_service(timeout_sec=2.):
                raise RuntimeError(f'MoveIt service unavailable: {prefix}/{name}')
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
            if not future.done():
                raise RuntimeError(f'MoveIt service timed out: {name}')
            result = future.result()
            if result is None:
                raise RuntimeError(f'MoveIt service returned no result: {name}')
            return result
        finally:
            node.destroy_client(client)

    def scene_read():
        request = GetPlanningScene.Request()
        request.components.components = 1 | 2 | 4 | 16 | 32 | 64 | 128 | 256
        return call(GetPlanningScene, 'get_planning_scene', request).scene

    def scene_digest(scene):
        value = message_to_ordereddict(scene)
        # Live joint states/timestamps are not part of static scene identity.
        value['robot_state'].pop('joint_state', None)
        value['robot_state'].pop('multi_dof_joint_state', None)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def fk(state):
        request = GetPositionFK.Request()
        request.header.frame_id = config['base_frame']
        request.fk_link_names = [config['flange_link']]
        request.robot_state = state
        result = call(GetPositionFK, 'compute_fk', request)
        if result.error_code.val != 1 or result.fk_link_names != [config['flange_link']] or len(result.pose_stamped) != 1:
            raise RuntimeError('MoveIt FK failed')
        if result.pose_stamped[0].header.frame_id != config['base_frame']:
            raise RuntimeError('MoveIt FK returned a different coordinate frame')
        return pose_matrix(result.pose_stamped[0].pose)

    try:
        scene = scene_read()
        world = {obj.id for obj in scene.world.collision_objects if obj.primitives or obj.meshes or obj.planes}
        attached = {obj.object.id for obj in scene.robot_state.attached_collision_objects
                    if obj.object.primitives or obj.object.meshes}
        for key, present in [('required_world_objects', world), ('required_attached_objects', attached)]:
            missing = set(config[key])-present
            if missing:
                raise ValueError(f'missing scene geometry for {key}: {sorted(missing)}')
        # Do not accept a scene that globally exempts a required obstacle.
        acm = scene.allowed_collision_matrix
        obstacles = set(config['required_world_objects'])
        for name, enabled in zip(acm.default_entry_names, acm.default_entry_values):
            if name in obstacles and enabled:
                raise ValueError(f'required obstacle has default allowed collisions: {name}')
        for name, row in zip(acm.entry_names, acm.entry_values):
            if name in obstacles and any(row.enabled):
                raise ValueError(f'required obstacle has allowed collision pairs: {name}')
        state = copy.deepcopy(scene.robot_state)
        state.is_diff = False
        joint_map = dict(zip(state.joint_state.name, state.joint_state.position))
        joint_map.update(zip(config['joint_names'], start_q))
        state.joint_state.name = list(joint_map)
        state.joint_state.position = [float(v) for v in joint_map.values()]
        state.joint_state.velocity = []
        state.joint_state.effort = []
        start_check = check_fk(poses[0], fk(state))
        request = GetCartesianPath.Request()
        request.header.frame_id = config['base_frame']
        request.start_state = state
        request.group_name = config['group']
        request.link_name = config['flange_link']
        request.max_step = .005
        request.jump_threshold = 2.
        request.revolute_jump_threshold = .10
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = .05
        request.max_acceleration_scaling_factor = .05
        request.cartesian_speed_limited_link = config['flange_link']
        request.max_cartesian_speed = .010
        for transform in poses[1:]:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, transform[:3, 3])
            quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
            request.waypoints.append(pose)
        result = call(GetCartesianPath, 'compute_cartesian_path', request)
        if result.error_code.val != 1 or not np.isfinite(result.fraction) or result.fraction < 1.-1e-9:
            raise ValueError(f'incomplete Cartesian path: code={result.error_code.val}, fraction={result.fraction}')
        trajectory = result.solution.joint_trajectory
        positions = [p.positions for p in trajectory.points]
        times = [p.time_from_start.sec+p.time_from_start.nanosec*1e-9 for p in trajectory.points]
        stats = check_trajectory(trajectory.joint_names, positions, times, config['joint_names'], start_q)
        samples = 0
        for q in dense_joint_samples(positions):
            lookup = dict(zip(config['joint_names'], q))
            state.joint_state.position = [float(lookup.get(name, value)) for name, value in
                                         zip(state.joint_state.name, state.joint_state.position)]
            validity = GetStateValidity.Request()
            validity.robot_state = state
            validity.group_name = config['group']
            if not call(GetStateValidity, 'check_state_validity', validity).valid:
                raise ValueError(f'invalid/colliding joint state at audit sample {samples}')
            samples += 1
        end_check = check_fk(poses[-1], fk(state))
        if scene_digest(scene) != scene_digest(scene_read()):
            raise ValueError('planning scene changed during audit; repeat')
        return dict(status='sampled_scene_audit_passed', executable=False, motion_started=False,
                    scene_sha256=scene_digest(scene), config=config,
                    source_candidate_sha256=hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest(),
                    plan_sha256=hashlib.sha256(Path(candidate['plan']).read_bytes()).hexdigest(),
                    snapshot_sha256=hashlib.sha256(Path(report['snapshot']).read_bytes()).hexdigest(),
                    start_fk=start_check, end_fk=end_check, trajectory_summary=stats,
                    collision_audit_samples=samples,
                    trajectory=message_to_ordereddict(result.solution),
                    limitations=['saved snapshot only; live state must be rechecked',
                                 'scene geometry has not been physically verified by this program',
                                 'sampled collision checks are not continuous swept-volume verification',
                                 'no trajectory execution integration; no wipe or return path audit'] +
                                (depth_scene['limitations'] if depth_scene is not None else []))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new audit file')
    try:
        result = audit(args.report, json.loads(args.config.read_text()))
    except Exception as exc:
        result = dict(status='blocked', executable=False, motion_started=False,
                      error=str(exc), error_type=type(exc).__name__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != 'trajectory'}, indent=2))
    return 0 if result['status'] == 'sampled_scene_audit_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
