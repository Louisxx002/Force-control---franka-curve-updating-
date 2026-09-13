"""Load measured boxes into the isolated planning service; never moves hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
from .approach_checks import validate_config
from .execute import rigid_transform, ExecutionError


def validate_scene(scene, config, attachments_only=False):
    validate_config(config)
    if scene.get('units') != 'm' or scene.get('measurements_confirmed') is not True:
        raise ValueError('scene requires confirmed measurements in metres')
    ids = set()
    for category, required, frame in [('world', 'required_world_objects', config['base_frame']),
                                      ('attached', 'required_attached_objects', config['flange_link'])]:
        if attachments_only and category == 'world':
            continue
        objects = scene.get(category, [])
        present = set()
        for obj in objects:
            name = obj['id']
            if not isinstance(name, str) or not name or name in ids:
                raise ValueError('object IDs must be unique nonempty strings')
            ids.add(name); present.add(name)
            if obj.get('frame') != frame:
                raise ValueError('unexpected object frame: '+name)
            dims = np.asarray(obj.get('size_m'), dtype=float)
            if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
                raise ValueError('box requires three positive measured dimensions: '+name)
            try:
                rigid_transform(obj.get('T_frame_box'), name)
            except ExecutionError as exc:
                raise ValueError(str(exc)) from exc
        if set(config[required])-present:
            raise ValueError('missing required objects: '+str(sorted(set(config[required])-present)))
    if config.get('namespace') != '/autowipe':
        raise ValueError('scene loader restricted to /autowipe planning service')
    return scene


def load(scene, config, attachments_only=False):
    validate_scene(scene, config, attachments_only)
    import rclpy
    from rclpy.node import Node
    from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
    from moveit_msgs.srv import ApplyPlanningScene
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation
    request = ApplyPlanningScene.Request()
    request.scene.is_diff = True
    request.scene.robot_state.is_diff = True
    for category in (('attached',) if attachments_only else ('world', 'attached')):
        for obj in scene[category]:
            collision = CollisionObject(id=obj['id'])
            collision.header.frame_id = obj['frame']
            collision.operation = CollisionObject.ADD
            transform = np.asarray(obj['T_frame_box'], float)
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, transform[:3, 3])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, Rotation.from_matrix(transform[:3, :3]).as_quat())
            collision.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(map(float, obj['size_m'])))]
            collision.primitive_poses = [pose]
            if category == 'world': request.scene.world.collision_objects.append(collision)
            else:
                attachment = AttachedCollisionObject(link_name=config['flange_link'], object=collision)
                # Only the mount link is exempt, never the rest of the arm.
                attachment.touch_links = [config['flange_link']]
                request.scene.robot_state.attached_collision_objects.append(attachment)
    rclpy.init(); node = Node('autowipe_load_measured_scene')
    try:
        client = node.create_client(ApplyPlanningScene, '/autowipe/apply_planning_scene')
        if not client.wait_for_service(timeout_sec=3.): raise RuntimeError('planning service unavailable')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('scene application failed or timed out; inspect scene before retry')
        return dict(status='scene_applied_requires_audit', executable=False, motion_started=False)
    finally:
        node.destroy_node(); rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attachments-only', action='store_true', help='load tool envelopes while retaining depth world geometry')
    args = parser.parse_args()
    try:
        result = load(json.loads(args.scene.read_text()), json.loads(args.config.read_text()), args.attachments_only)
    except Exception as exc:
        parser.exit(1, str(exc)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
