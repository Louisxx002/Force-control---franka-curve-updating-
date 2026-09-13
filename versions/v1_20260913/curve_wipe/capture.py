"""Static RGB/XYZ capture. Robot access is readOnce only; no motion methods."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_state(ip):
    start = time.monotonic()
    result = subprocess.run([str(ROOT / 'read_state'), ip], capture_output=True,
                            text=True, timeout=8, check=True)
    state = json.loads(result.stdout)
    state['host_start_monotonic_s'] = start
    state['host_end_monotonic_s'] = time.monotonic()
    return state


def rasterize_xyz(vertices_depth, texture_uv, T_rgb_depth, height, width):
    """Z-buffer SDK depth points into RGB pixels, keeping their RGB-frame XYZ."""
    pts = np.asarray(vertices_depth, dtype=float)
    tex = np.asarray(texture_uv, dtype=float)
    valid = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0) & np.isfinite(tex).all(axis=1)
    pts, tex = pts[valid], tex[valid]
    xyz = pts @ T_rgb_depth[:3, :3].T + T_rgb_depth[:3, 3]
    uv = np.floor(tex * [width, height] + 0.5).astype(int)
    keep = (xyz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    xyz, uv = xyz[keep], uv[keep]
    order = np.argsort(xyz[:, 2], kind='stable')
    linear = uv[order, 1] * width + uv[order, 0]
    _, first = np.unique(linear, return_index=True)
    out = np.full((height * width, 3), np.nan, dtype=np.float32)
    out[linear[first]] = xyz[order[first]]
    return out.reshape(height, width, 3)


def capture(output, serial='243722072895', robot_ip=None):
    import pyrealsense2 as rs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    started = False
    try:
        pipeline.start(cfg)
        started = True
        for _ in range(25):
            pipeline.wait_for_frames(5000)
        before = read_state(robot_ip) if robot_ip else None
        host_start = time.monotonic()
        frames = pipeline.wait_for_frames(5000)
        host_end = time.monotonic()
        after = read_state(robot_ip) if robot_ip else None
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        bgr = np.asanyarray(color.get_data()).copy()
        pc = rs.pointcloud()
        pc.map_to(color)
        points = pc.calculate(depth)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3).copy()
        uv = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2).copy()
        ext = depth.profile.get_extrinsics_to(color.profile)
        T = np.eye(4)
        T[:3, :3] = np.asarray(ext.rotation).reshape(3, 3, order='F')
        T[:3, 3] = ext.translation
        xyz = rasterize_xyz(verts, uv, T, *bgr.shape[:2])
        # SDK alignment rasterizes the depth pixel footprint into the RGB
        # image, avoiding the one-pixel gaps of nearest-point splatting.
        # Aligned depth has RGB intrinsics; pointcloud XYZ is in that frame.
        aligned_depth = rs.align(rs.stream.color).process(frames).get_depth_frame()
        aligned_pc = rs.pointcloud()
        aligned_points = aligned_pc.calculate(aligned_depth)
        xyz = np.asanyarray(aligned_points.get_vertices()).view(np.float32).reshape(*bgr.shape[:2], 3).copy()
        xyz[xyz[..., 2] <= 0] = np.nan
        intr = color.profile.as_video_stream_profile().intrinsics
        metadata = dict(schema_version=1, source='live_realsense', camera_serial=serial,
                        xyz_frame='C_LRGB', xyz_units='metres', synthetic=False,
                        projection='SDK depth aligned to RGB, then pointcloud with aligned intrinsics; no hole fill',
                        host_start_monotonic_s=host_start, host_end_monotonic_s=host_end,
                        color_timestamp_ms=color.get_timestamp(), depth_timestamp_ms=depth.get_timestamp(),
                        timestamp_domain=str(color.get_frame_timestamp_domain()),
                        rgb_intrinsics=dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                                            width=intr.width, height=intr.height, distortion=str(intr.model), coeffs=intr.coeffs),
                        robot_before=before, robot_after=after, static_pose_validated=False)
        if before is not None:
            a = np.asarray(before['O_T_EE']).reshape(4, 4, order='F')
            b = np.asarray(after['O_T_EE']).reshape(4, 4, order='F')
            translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
            rotation = float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))))
            max_dq = float(np.max(np.abs([before['dq'], after['dq']])))
            metadata.update(T_base_ee=a.tolist(), pose_delta_m=translation, pose_delta_deg=rotation,
                            max_joint_speed_rad_s=max_dq,
                            static_pose_validated=translation < .0005 and rotation < .2 and max_dq < .01,
                            timing_note='Static bracketing check only; not exposure-synchronized robot measurements.')
        np.savez_compressed(output / 'snapshot.npz', bgr=bgr, xyz_rgb_m=xyz,
                            metadata_json=json.dumps(metadata, ensure_ascii=False))
        cv2.imwrite(str(output / 'color.png'), bgr)
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(json.dumps(dict(output=str(output), valid_xyz_pixels=int(np.isfinite(xyz).all(axis=2).sum()),
                              static_pose_validated=metadata['static_pose_validated']), indent=2))
        return output
    finally:
        if started:
            pipeline.stop()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--serial', default='243722072895')
    p.add_argument('--robot-ip')
    args = p.parse_args()
    capture(args.output, args.serial, args.robot_ip)
