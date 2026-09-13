"""Probe real MoveIt services and saved-state FK without moving hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
from .approach_checks import validate_config, check_fk
from .check_approach import pose_matrix


def probe(snapshot, config):
    validate_config(config)
    with np.load(snapshot, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
    if metadata.get('synthetic') is not False or not metadata.get('static_pose_validated'):
        raise ValueError('requires a real statically validated snapshot')
    import rclpy
    from rclpy.node import Node
    from moveit_msgs.srv import GetPositionFK, GetPlanningScene, GetStateValidity
    rclpy.init()
    node = Node('autowipe_planning_probe')
    prefix = config.get('namespace', '').rstrip('/')
    def call(kind, name, request):
        client = node.create_client(kind, prefix+'/'+name)
        try:
            if not client.wait_for_service(timeout_sec=3.):
                raise RuntimeError('unavailable service: '+name)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
            if not future.done() or future.result() is None:
                raise RuntimeError('service failed or timed out: '+name)
            return future.result()
        finally:
            node.destroy_client(client)
    try:
        request = GetPlanningScene.Request(); request.components.components = 1023
        scene = call(GetPlanningScene, 'get_planning_scene', request).scene
        request = GetPositionFK.Request()
        request.header.frame_id = config['base_frame']
        request.fk_link_names = [config['flange_link']]
        request.robot_state.joint_state.name = config['joint_names']
        request.robot_state.joint_state.position = list(map(float, metadata['robot_before']['q']))
        response = call(GetPositionFK, 'compute_fk', request)
        if response.error_code.val != 1 or response.fk_link_names != request.fk_link_names or len(response.pose_stamped) != 1:
            raise ValueError('FK service failed')
        if response.pose_stamped[0].header.frame_id != config['base_frame']:
            raise ValueError('FK frame mismatch')
        flange = np.array(metadata['robot_before']['F_T_EE']).reshape(4, 4, order='F')
        expected = np.array(metadata['T_base_ee']) @ np.linalg.inv(flange)
        fk = check_fk(expected, pose_matrix(response.pose_stamped[0].pose))
        validity = GetStateValidity.Request()
        validity.group_name = config['group']; validity.robot_state = request.robot_state
        valid = call(GetStateValidity, 'check_state_validity', validity)
        world = {o.id for o in scene.world.collision_objects if o.primitives or o.meshes or o.planes}
        attached = {o.object.id for o in scene.robot_state.attached_collision_objects if o.object.primitives or o.object.meshes}
        return dict(status='model_services_verified', executable=False, motion_started=False,
                    snapshot=str(Path(snapshot).resolve()), config=config, start_fk=fk,
                    saved_state_valid_in_current_scene=valid.valid,
                    missing_world_objects=sorted(set(config['required_world_objects'])-world),
                    missing_attached_objects=sorted(set(config['required_attached_objects'])-attached),
                    limitations=['saved state only; not current hardware state',
                                 'model match does not verify physical collision geometry'])
    finally:
        node.destroy_node(); rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    try:
        result = probe(args.snapshot, json.loads(args.config.read_text()))
    except Exception as exc:
        result = dict(status='blocked', executable=False, motion_started=False, error=str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))
    return int(result['status'] == 'blocked')

if __name__ == '__main__':
    raise SystemExit(main())
