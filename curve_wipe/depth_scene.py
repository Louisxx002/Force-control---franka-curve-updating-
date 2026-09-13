"""RealSense snapshot -> observed voxel collision geometry, without environment CAD.

Only occupied observations are represented. Unknown space is NOT certified free.
No execution is provided by this module.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import _transform

ROOT = Path(__file__).resolve().parents[1]


def voxelize(xyz, transform, voxel_m=.02, padding_m=.01, max_voxels=60000):
    if not np.isfinite(voxel_m) or not .005 <= voxel_m <= .05:
        raise ValueError('voxel size must be between 5 and 50 mm')
    if not np.isfinite(padding_m) or not 0 <= padding_m <= .05:
        raise ValueError('padding must be between 0 and 50 mm')
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError('expected H x W x 3 RGB-frame points')
    transform = _transform(transform, 'T_base_camera')
    mask = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    points = xyz[mask]
    if not len(points): raise ValueError('no valid observed depth')
    points = points @ transform[:3, :3].T + transform[:3, 3]
    if np.max(np.abs(points)) > 100:
        raise ValueError('implausible point coordinates; check depth units')
    indices = np.unique(np.floor(points/voxel_m).astype(np.int64), axis=0)
    if len(indices) > max_voxels:
        raise ValueError(f'{len(indices)} voxels exceeds {max_voxels}; no observations silently dropped')
    centers = (indices+.5)*voxel_m
    return centers, dict(valid_points=len(points), invalid_pixels=int(mask.size-mask.sum()),
                         voxel_count=len(centers), voxel_m=voxel_m, padding_m=padding_m,
                         box_size_m=voxel_m+2*padding_m)


def build(snapshot, handeye, output, voxel_m=.02, padding_m=.01):
    snapshot, handeye, output = map(Path, (snapshot, handeye, output))
    if output.exists(): raise ValueError('output directory already exists')
    with np.load(snapshot, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json'])); xyz = data['xyz_rgb_m'].copy()
    calibration = json.loads(handeye.read_text())
    if meta.get('synthetic') is not False or not meta.get('static_pose_validated'):
        raise ValueError('requires real snapshot with validated static pose')
    if meta.get('xyz_frame') != 'C_LRGB' or meta.get('xyz_units') != 'metres':
        raise ValueError('expected RGB optical-frame points in metres')
    if meta.get('camera_serial') != calibration['camera_serial']:
        raise ValueError('camera differs from handeye calibration')
    tec = _transform(calibration['T_EE_L_C_LRGB'], 'T_ee_camera')
    if not np.allclose(tec @ _transform(calibration['T_C_LRGB_EE_L'], 'T_camera_ee'), np.eye(4), atol=1e-5):
        raise ValueError('handeye inverse mismatch')
    tbc = _transform(meta['T_base_ee'], 'T_base_ee') @ tec
    centers, stats = voxelize(xyz, tbc, voxel_m, padding_m)
    snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    calibration_hash = hashlib.sha256(handeye.read_bytes()).hexdigest()
    geometry_hash = hashlib.sha256(centers.astype('<f8').tobytes()+np.float64(stats['box_size_m']).tobytes()).hexdigest()
    object_id = 'realsense_observed_'+geometry_hash[:20]
    manifest = dict(status='observed_geometry_only', executable=False, motion_started=False,
                    snapshot=str(snapshot.resolve()), snapshot_sha256=snapshot_hash,
                    handeye_sha256=calibration_hash, geometry_sha256=geometry_hash,
                    object_id=object_id, base_frame='fr3_link0', T_base_camera=tbc.tolist(), **stats,
                    limitations=['occupied observations only; unknown/occluded space is not certified free',
                                 'historical snapshot is not a live environment update',
                                 'no robot self-filter; robot observations can cause conservative rejection',
                                 'tool envelope and calibration mounting still require verification'])
    config = json.loads((ROOT/'config/motion_scene.local.json').read_text())
    config.update(required_world_objects=[object_id], observed_depth_scene=manifest,
                  note='No environment CAD. Observed depth voxels only; attachment geometry still required.')
    output.mkdir(parents=True)
    np.savez_compressed(output/'voxels.npz', centers=centers)
    (output/'scene.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'motion_config.json').write_text(json.dumps(config, indent=2)+'\n')
    return manifest, centers


def apply(manifest, centers):
    import rclpy
    from rclpy.node import Node
    from moveit_msgs.msg import CollisionObject
    from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
    from shape_msgs.msg import SolidPrimitive
    from geometry_msgs.msg import Pose
    from tf2_ros import Buffer, TransformListener
    from .check_approach import pose_matrix
    rclpy.init(); node = Node('autowipe_depth_scene')
    buffer = Buffer(); listener = TransformListener(buffer, node)
    def call(kind, name, request):
        client = node.create_client(kind, '/autowipe/'+name)
        try:
            if not client.wait_for_service(timeout_sec=3.): raise RuntimeError('service unavailable: '+name)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=20.)
            if not future.done() or future.result() is None: raise RuntimeError('service timeout: '+name)
            return future.result()
        finally:
            node.destroy_client(client)
    try:
        obj = CollisionObject(id=manifest['object_id'])
        obj.header.frame_id = manifest['base_frame']; obj.operation = CollisionObject.ADD
        size = float(manifest['box_size_m'])
        for center in centers:
            pose = Pose(); pose.orientation.w = 1.
            pose.position.x, pose.position.y, pose.position.z = map(float, center)
            obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[size]*3))
            obj.primitive_poses.append(pose)
        request = ApplyPlanningScene.Request(); request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        request.scene.world.collision_objects = [obj]
        # Retain earlier observations and existing obstacles; do not clear unseen areas.
        if not call(ApplyPlanningScene, 'apply_planning_scene', request).success:
            raise RuntimeError('scene application rejected')
        read = GetPlanningScene.Request(); read.components.components = 16
        scene = call(GetPlanningScene, 'get_planning_scene', read).scene
        matches = [x for x in scene.world.collision_objects if x.id == obj.id]
        if len(matches) != 1 or len(matches[0].primitives) != len(centers):
            raise RuntimeError('scene readback voxel count mismatch')
        actual = matches[0]
        frame_transform = np.eye(4)
        if actual.header.frame_id != obj.header.frame_id:
            future = buffer.wait_for_transform_async(obj.header.frame_id, actual.header.frame_id, rclpy.time.Time())
            rclpy.spin_until_future_complete(node, future, timeout_sec=3.)
            if not future.done(): raise RuntimeError('scene readback transform unavailable')
            tf = future.result().transform
            pose = Pose(); pose.position.x = tf.translation.x; pose.position.y = tf.translation.y; pose.position.z = tf.translation.z
            pose.orientation = tf.rotation
            frame_transform = pose_matrix(pose)
        object_transform = frame_transform @ pose_matrix(actual.pose)
        actual_poses = [object_transform @ pose_matrix(p) for p in actual.primitive_poses]
        if len(actual_poses) != len(centers) or any(not np.allclose(p[:3,:3],np.eye(3),rtol=0,atol=1e-10) for p in actual_poses):
            raise RuntimeError('scene readback orientations mismatch')
        actual_centers = [p[:3,3] for p in actual_poses]
        if not np.allclose(actual_centers, centers, rtol=0, atol=1e-10):
            raise RuntimeError('scene readback positions mismatch')
        if any(p.type != SolidPrimitive.BOX or not np.allclose(p.dimensions,[size]*3,rtol=0,atol=1e-10) for p in actual.primitives):
            raise RuntimeError('scene readback sizes mismatch')
        return dict(status='observed_voxels_applied_and_read_back', voxel_count=len(centers),
                    executable=False, motion_started=False, object_id=obj.id)
    finally:
        node.destroy_node(); rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--handeye', type=Path, default=ROOT/'config/handeye_result.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--voxel-m', type=float, default=.02)
    parser.add_argument('--padding-m', type=float, default=.01)
    parser.add_argument('--apply', action='store_true', help='apply to isolated /autowipe service and verify readback')
    args = parser.parse_args()
    try:
        manifest, centers = build(args.snapshot, args.handeye, args.output, args.voxel_m, args.padding_m)
        result = apply(manifest, centers) if args.apply else manifest
        if args.apply: (args.output/'apply_result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    except Exception as exc:
        parser.exit(1, str(exc)+'\n')

if __name__ == '__main__': main()
