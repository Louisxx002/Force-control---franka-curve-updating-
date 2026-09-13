"""Pure validation for read-only MoveIt approach audits; never grants execution."""
import numpy as np
from .execute import rigid_transform, rotation_angle_deg


def validate_config(config):
    required = ('group', 'base_frame', 'flange_link', 'joint_names',
                'required_world_objects', 'required_attached_objects')
    for key in required:
        if not config.get(key):
            raise ValueError(f'missing planning configuration: {key}')
    names = config['joint_names']
    if len(names) != 7 or len(set(names)) != 7 or not all(isinstance(x, str) and x for x in names):
        raise ValueError('joint_names must contain seven distinct names in libfranka q order')
    for key in ('required_world_objects', 'required_attached_objects'):
        if not isinstance(config[key], list) or not all(isinstance(x, str) and x for x in config[key]):
            raise ValueError(f'{key} must be a nonempty list of object IDs')
    return config


def flange_waypoints(ee_poses, flange_to_ee):
    transform = rigid_transform(flange_to_ee, 'F_T_EE')
    return [rigid_transform(p, 'EE waypoint') @ np.linalg.inv(transform) for p in ee_poses]


def check_fk(expected, actual):
    expected, actual = rigid_transform(expected, 'expected pose'), rigid_transform(actual, 'FK pose')
    distance = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
    angle = rotation_angle_deg(expected[:3, :3], actual[:3, :3])
    if distance > .002 or angle > .5:
        raise ValueError(f'robot model/FK mismatch: {distance*1000:.2f} mm, {angle:.2f} deg')
    return dict(position_error_m=distance, orientation_error_deg=angle)


def check_trajectory(names, positions, times, expected_names, start_q):
    if list(names) != list(expected_names):
        raise ValueError('trajectory joint names/order differ from configured libfranka order')
    q, t = np.asarray(positions, float), np.asarray(times, float)
    start = np.asarray(start_q, float)
    if (q.ndim != 2 or q.shape[1:] != (7,) or len(q) < 2 or t.shape != (len(q),)
            or start.shape != (7,) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or not np.isfinite(start).all()):
        raise ValueError('invalid or nonfinite joint trajectory')
    if t[0] < 0 or np.any(np.diff(t) <= 0):
        raise ValueError('trajectory timestamps must strictly increase')
    if np.max(np.abs(q[0]-start)) > .001:
        raise ValueError('trajectory starts at a different joint state')
    if np.max(np.abs(np.diff(q, axis=0))) > .10:
        raise ValueError('trajectory contains a joint step larger than 0.10 rad')
    if np.max(q[:, 1]) >= 1.72:
        raise ValueError('trajectory crosses existing joint-2 guard')
    return dict(points=len(q), duration_s=float(t[-1]),
                maximum_joint_step_rad=float(np.max(np.abs(np.diff(q, axis=0)))))


def dense_joint_samples(positions, step=.01):
    """Discrete audit samples only; does not prove continuous swept-volume clearance."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be positive')
    q = np.asarray(positions, float)
    if q.ndim != 2 or q.shape[1:] != (7,) or not len(q) or not np.isfinite(q).all():
        raise ValueError('invalid joint samples')
    yield q[0]
    for a, b in zip(q, q[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(b-a))/step)))
        for index in range(1, count+1):
            yield a+(b-a)*index/count
