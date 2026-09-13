"""One short surface-normal-aligned wipe. Import and preflight never open hardware.

Run with --execute only after the real plan has been reviewed. Force tracking
assumes sensor +Z is coaxial with EE +Z; it uses signed -delta Fz, not a force
magnitude. Free-space tare is taken before contact; during wiping the measured
surface normal determines the TCP/EE orientation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

PERIOD = 0.010
# Python updates targets at 100 Hz; Franky/libfranka runs the hardware loop.
# Wall-clock delays are checked here, but must not become a catch-up motion.
MAX_LOOP_DT = 0.12
STANDOFF = 0.050
SEARCH_START = 0.015
MAX_PENETRATION = 0.012
TARGET_FORCE = 1.0
FREE_SPACE_FORCE_LIMIT_N = 0.75
CONTACT_DETECT_FORCE_N = 0.30
CONTACT_CONFIRM_S = 1.0
CONTACT_PRESENT_FORCE_N = 0.20
FORCE_SETTLE_TOLERANCE_N = 0.25
TOTAL_FORCE_LIMIT_N = 8.0
NORMAL_FORCE_LIMIT_N = 8.0
# Permit up to 1.5 N of tensile/reverse reaction before stopping.
MAX_TENSILE_NORMAL_N = 1.5
FORCE_LOSS_TIMEOUT_S = 2.0
ADMITTANCE_GAIN_M_PER_N_S = 0.0015
MAX_ADMITTANCE_RATE_M_S = 0.002
MAX_SEGMENT_LENGTH = 0.200
DEFAULT_MAX_APPROACH_M = 0.500
DEFAULT_MAX_SURFACE_HEIGHT_M = 0.020
DEFAULT_MAX_NORMAL_ANGLE_DEG = 25.0
NORMAL_SMOOTHING_SIGMA_M = 0.006
WIPE_RAMP_TIME_S = 0.5


def validate_approach_limit(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('maximum approach distance must be finite and positive')
    return value


def check_approach_distance(start, entry, max_approach_m=DEFAULT_MAX_APPROACH_M):
    limit = validate_approach_limit(max_approach_m)
    distance = float(np.linalg.norm(np.asarray(entry)-np.asarray(start)))
    if not np.isfinite(distance) or distance > limit:
        raise ExecutionError(f'travel to precontact is {distance*1000:.1f} mm; limit is {limit*1000:g} mm')
    return distance


def pass_distance(elapsed, duration, length, reverse=False):
    """Constant-speed pass with jerk-limited 0.5 s entry and exit ramps."""
    if not np.isfinite([elapsed, duration, length]).all() or duration <= 0 or length < 0:
        raise ValueError("invalid pass timing")
    t = float(np.clip(elapsed, 0.0, duration))
    ramp = min(WIPE_RAMP_TIME_S, duration / 2.0)
    cruise = max(0.0, duration - 2.0 * ramp)
    # The cubic ramp has zero velocity/acceleration at its start and zero
    # acceleration at its join with the constant-speed section.
    speed = length / max(duration - ramp, 1e-12)
    if t < ramp:
        u = t / ramp
        distance = speed * ramp * (u**3 - 0.5*u**4)
    elif t <= ramp + cruise:
        distance = 0.5 * speed * ramp + speed * (t - ramp)
    else:
        u = (duration - t) / ramp
        distance = length - speed * ramp * (u**3 - 0.5*u**4)
    return length - distance if reverse else distance


class ExecutionError(RuntimeError):
    pass


def active_error_names(errors):
    # pybind Errors has no Python __bool__: even an empty Errors() is truthy.
    return sorted(name for name in dir(errors) if not name.startswith("_")
                  and isinstance(getattr(errors, name), (bool, np.bool_))
                  and bool(getattr(errors, name)))


def check_robot_state(state, allowed_modes, log=None, phase=None):
    """Use one snapshot for both mode and faults; never sample has_errors again."""
    mode = getattr(state.robot_mode, "value", state.robot_mode)
    errors = active_error_names(state.current_errors)
    if errors or mode not in allowed_modes:
        fault = dict(mode=str(state.robot_mode), current_errors=errors,
                     last_motion_errors=active_error_names(state.last_motion_errors),
                     control_command_success_rate=float(state.control_command_success_rate),
                     phase=phase)
        if log is not None:
            log["robot_fault"] = fault
        raise ExecutionError("robot fault: " + json.dumps(fault, ensure_ascii=False))


def motion_step(dt):
    """A late Python tick slows progress instead of issuing a catch-up jump."""
    if not np.isfinite(dt) or not 0 < dt <= MAX_LOOP_DT:
        raise ExecutionError(f"control loop stalled: dt={dt} s")
    return min(dt, PERIOD)


class NormalAdmittance:
    """First-order normal velocity response, continuous across contact phases."""
    def __init__(self, velocity=0.0):
        self.velocity = float(velocity)

    def step(self, height, force, dt):
        dt = motion_step(dt)
        target = float(np.clip(ADMITTANCE_GAIN_M_PER_N_S * (force - TARGET_FORCE),
                               -MAX_ADMITTANCE_RATE_M_S, MAX_ADMITTANCE_RATE_M_S))
        tau = 0.2
        decay = math.exp(-dt / tau)
        displacement = target * dt + (self.velocity - target) * tau * (1 - decay)
        self.velocity = target + (self.velocity - target) * decay
        result = height + displacement
        if result < -MAX_PENETRATION:
            raise ExecutionError("normal penetration exceeds 12 mm below visual surface")
        return result


def rigid_transform(value, name):
    a = np.asarray(value, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all() or
            not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0) or
            not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5, rtol=0) or
            not np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ExecutionError(f"{name}: expected a finite rigid 4 x 4 transform")
    return a


def rotation_angle_deg(a, b):
    # atan2 is stable for float32 robot matrices close to identity.
    r = a @ b.T
    sine = np.linalg.norm([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0],
                           r[1, 0] - r[0, 1]]) / 2
    return float(np.degrees(np.arctan2(sine, (np.trace(r) - 1) / 2)))


def prepare_plan(plan, segment=None, *, require_real=False,
                 max_surface_height_m=DEFAULT_MAX_SURFACE_HEIGHT_M,
                 max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG):
    if not np.isfinite(max_surface_height_m) or max_surface_height_m <= 0:
        raise ValueError('surface height limit must be finite and positive')
    if not np.isfinite(max_normal_angle_deg) or not 0 < max_normal_angle_deg < 90:
        raise ValueError('normal angle limit must be finite and between 0 and 90 degrees')
    metadata = plan.get("metadata", {})
    synthetic = metadata.get("synthetic", plan.get("synthetic"))
    candidate = metadata.get("executable_candidate", plan.get("executable_candidate"))
    if require_real and (synthetic is not False or candidate is not True):
        raise ExecutionError("execution requires synthetic=false and executable_candidate=true")
    capture = rigid_transform(metadata.get("T_base_ee_capture"), "T_base_ee_capture")
    tool = rigid_transform(metadata.get("T_ee_tcp"), "T_ee_tcp")
    flange = rigid_transform(metadata.get("F_T_EE_capture"), "F_T_EE_capture")
    if np.linalg.norm(tool[:3, 3]) > 0.25:
        raise ExecutionError("TCP offset exceeds 250 mm; check units")
    if not np.allclose(tool[:3, 2], [0, 0, 1], atol=1e-5, rtol=0):
        raise ExecutionError("T_ee_tcp must keep TCP +Z parallel to EE +Z")
    rotation = capture[:3, :3].copy()
    normal = -rotation[:, 2] / np.linalg.norm(rotation[:, 2])
    segments = plan.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ExecutionError("plan contains no segments")
    if segment is not None and (segment < 0 or segment >= len(segments)):
        raise ExecutionError("segment index out of range (indices start at zero)")
    candidates, reasons = [], []
    for index, item in enumerate(segments):
        if segment is not None and index != segment:
            continue
        try:
            waypoints = item["waypoints"]
            points = np.asarray([w["surface_point_base_m"] for w in waypoints], float)
            raw_normals = np.asarray([w["normal_out_base"] for w in waypoints], float)
            if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or
                    raw_normals.shape != points.shape or not np.isfinite(points).all() or
                    not np.isfinite(raw_normals).all()):
                raise ExecutionError("invalid points/normals")
            if not np.allclose(np.linalg.norm(raw_normals, axis=1), 1, atol=1e-3, rtol=0):
                raise ExecutionError("normals must be unit vectors")
            raw_normals /= np.linalg.norm(raw_normals, axis=1)[:, None]
            angles = np.degrees(np.arccos(np.clip(raw_normals @ normal, -1, 1)))
            tcp_frames = []
            for waypoint in waypoints:
                raw_frame = waypoint.get("T_base_tcp_contact")
                if raw_frame is None:
                    tcp_frames = []
                    break
                frame = rigid_transform(raw_frame, "T_base_tcp_contact")
                tcp_frames.append(frame[:3, :3])
            spatial = len(tcp_frames) == len(waypoints)
            if not spatial and np.max(angles) >= max_normal_angle_deg:
                raise ExecutionError(f"surface normal differs from held tool normal by >={max_normal_angle_deg:g} degrees")
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if np.any(lengths < 1e-7):
                raise ExecutionError("duplicate/degenerate consecutive waypoints")
            arc = np.r_[0.0, np.cumsum(lengths)]
            max_segment_length = float(plan.get("metadata", {}).get(
                "max_segment_length_m", MAX_SEGMENT_LENGTH))
            if not np.isfinite(max_segment_length) or max_segment_length <= 0:
                raise ExecutionError("invalid maximum segment length")
            if not 0.001 <= arc[-1] <= max_segment_length + 1e-10:
                raise ExecutionError(f"segment length must be 1..{max_segment_length*1000:g} mm")
            if spatial:
                tcp_frames = np.asarray(tcp_frames)
                # Plan a smooth normal trajectory once, before motion.  This
                # filters depth/PCA noise while retaining the measured surface
                # direction and prevents a large orientation step from being
                # created by one noisy waypoint.
                weights = np.exp(-0.5 * ((arc[:, None] - arc[None, :]) /
                                         NORMAL_SMOOTHING_SIGMA_M) ** 2)
                normals = weights @ raw_normals
                normals /= np.linalg.norm(normals, axis=1)[:, None]
                smoothing_error = np.degrees(np.arccos(np.clip(
                    np.sum(normals * raw_normals, axis=1), -1, 1)))
                # The planner's TCP frame has +Z into the surface.  Convert it
                # to the commanded EE orientation through EE->TCP calibration.
                ee_frames = np.asarray([r @ tool[:3, :3].T for r in tcp_frames])
            else:
                normals = raw_normals
                smoothing_error = np.zeros(len(normals))
                ee_frames = np.repeat(rotation[None, ...], len(waypoints), axis=0)
            # These fixed-pose geometry limits are only relevant to the legacy
            # executor. A spatial plan follows its measured normals directly.
            if not spatial and np.ptp(points @ normal) > max_surface_height_m:
                raise ExecutionError(f"surface height variation exceeds {max_surface_height_m*1000:g} mm")
            candidates.append(dict(index=index, points=points, normals=normals,
                                   raw_normals=raw_normals,
                                   normal_smoothing_max_deg=float(smoothing_error.max()),
                                   tcp_frames=tcp_frames, ee_frames=ee_frames,
                                   spatial_orientation=spatial, arc=arc,
                                   length_m=float(arc[-1]), max_normal_angle_deg=float(angles.max())))
        except (KeyError, TypeError, ValueError, ExecutionError) as exc:
            reasons.append(f"segment {index}: {exc}")
    if not candidates:
        raise ExecutionError("no valid short segment; " + "; ".join(reasons))
    selected = max(candidates, key=lambda s: s["length_m"])
    if selected["spatial_orientation"]:
        selected["orientation_slerp"] = Slerp(selected["arc"], Rotation.from_matrix(selected["ee_frames"]))
    return {**selected, "max_surface_height_mm": max_surface_height_m*1000,
            "max_normal_angle_deg": max_normal_angle_deg, "rotation": rotation, "normal": normal,
            "tcp_offset": tool[:3, 3].copy(), "tool_rotation": tool[:3, :3].copy(),
            "capture": capture,
            "flange": flange, "synthetic": synthetic, "executable_candidate": candidate}


def ee_position(surface, height, rotation, tcp_offset, normal):
    return np.asarray(surface) + float(height) * normal - rotation @ tcp_offset


def surface_at(prepared, distance):
    return np.array([np.interp(distance, prepared["arc"], prepared["points"][:, k])
                     for k in range(3)])


def normal_at(prepared, distance):
    if "arc" not in prepared or "normals" not in prepared:
        return np.asarray(prepared["normal"], float)
    n = np.array([np.interp(distance, prepared["arc"], prepared["normals"][:, k])
                  for k in range(3)])
    return n / np.linalg.norm(n)


def rotation_at(prepared, distance):
    if not prepared.get("spatial_orientation"):
        return prepared["rotation"]
    # Rebuild the TCP frame from the measured surface normal at every point.
    # Slerping the complete EE pose can leave its Z axis several degrees away
    # from the interpolated normal, especially when the normal changes quickly.
    s = float(np.clip(distance, 0.0, prepared["arc"][-1]))
    z_tcp = -normal_at(prepared, s)
    z_tcp /= np.linalg.norm(z_tcp)

    # Keep the planner's longitudinal direction as the tangent reference, then
    # project it into the plane orthogonal to the current normal.  This avoids
    # roll jumps while making the Z-axis constraint exact.
    tcp_frames = prepared["tcp_frames"]
    x_ref = np.array([np.interp(s, prepared["arc"], tcp_frames[:, k, 0])
                      for k in range(3)])
    x_ref -= z_tcp * float(x_ref @ z_tcp)
    x_norm = np.linalg.norm(x_ref)
    if x_norm < 1e-8:
        # A degenerate tangent reference is unlikely for a valid strip, but a
        # deterministic fallback keeps the frame orthonormal on noisy data.
        basis = np.eye(3)[np.argmin(np.abs(np.eye(3) @ z_tcp))]
        x_ref = basis - z_tcp * float(basis @ z_tcp)
        x_norm = np.linalg.norm(x_ref)
    x_tcp = x_ref / x_norm
    y_tcp = np.cross(z_tcp, x_tcp)
    y_tcp /= np.linalg.norm(y_tcp)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp /= np.linalg.norm(x_tcp)
    tcp_rotation = np.column_stack((x_tcp, y_tcp, z_tcp))
    return tcp_rotation @ prepared["tool_rotation"].T


def validate_live_pose(prepared, pose, flange, max_approach_m=DEFAULT_MAX_APPROACH_M):
    pose = rigid_transform(pose, "current O_T_EE")
    flange = rigid_transform(flange, "current F_T_EE")
    if rotation_angle_deg(pose[:3, :3], prepared["rotation"]) >= 1:
        raise ExecutionError("current orientation differs from capture by >=1 degree; recapture")
    if not np.allclose(flange, prepared["flange"], atol=1e-6, rtol=0):
        raise ExecutionError("F_T_EE changed after capture; calibration is not applicable")
    tcp = pose[:3, 3] + pose[:3, :3] @ prepared["tcp_offset"]
    if np.min((tcp - prepared["points"]) @ prepared["normal"]) < 0.020:
        raise ExecutionError("current TCP must be >=20 mm above the local surface for free-space tare")
    entry_rotation = rotation_at(prepared, 0.0)
    entry_normal = normal_at(prepared, 0.0)
    entry = ee_position(prepared["points"][0], STANDOFF, entry_rotation,
                        prepared["tcp_offset"], entry_normal)
    check_approach_distance(pose[:3, 3], entry, max_approach_m)
    return entry


def tare_from_samples(samples):
    data = np.asarray(samples, float)
    if data.ndim != 2 or data.shape[1:] != (6,) or len(data) < 50 or not np.isfinite(data).all():
        raise ExecutionError("tare requires >=50 finite, new six-axis samples over one second")
    std = data[:, :3].std(axis=0)
    if np.any(std >= 0.1):
        raise ExecutionError(f"free-space force is not quiet: std={std.tolist()} N")
    return np.median(data, axis=0), std


def check_wrench(raw, tare, sign=-1.0, *, reject_negative=True):
    delta = np.asarray(raw, float) - np.asarray(tare, float)
    if delta.shape != (6,) or not np.isfinite(delta).all():
        raise ExecutionError("nonfinite or malformed sensor wrench")
    force = float(sign * delta[2])
    if np.linalg.norm(delta[:3]) > TOTAL_FORCE_LIMIT_N:
        raise ExecutionError(f"contact force magnitude exceeds {TOTAL_FORCE_LIMIT_N:g} N")
    if np.linalg.norm(delta[3:]) > 0.5:
        raise ExecutionError("contact torque magnitude exceeds 0.5 Nm")
    if force > NORMAL_FORCE_LIMIT_N:
        raise ExecutionError(f"normal force exceeds {NORMAL_FORCE_LIMIT_N:g} N")
    if reject_negative and force < -MAX_TENSILE_NORMAL_N:
        raise ExecutionError(f"negative reaction below -{MAX_TENSILE_NORMAL_N:g} N; check sensor force sign/mount")
    return force, delta


def update_height(height, force, dt):
    # Virtual normal admittance: force error becomes a bounded normal velocity.
    rate = float(np.clip(ADMITTANCE_GAIN_M_PER_N_S * (force - TARGET_FORCE),
                         -MAX_ADMITTANCE_RATE_M_S, MAX_ADMITTANCE_RATE_M_S))
    result = height + rate * dt
    if result < -MAX_PENETRATION:
        raise ExecutionError("normal penetration exceeds 12 mm below visual surface")
    return result


def print_preflight(prepared, max_approach_m=DEFAULT_MAX_APPROACH_M):
    print(json.dumps({"mode": "preflight_only_no_hardware", "segment_index": prepared["index"],
                      "length_mm": prepared["length_m"] * 1000,
                      "normal_angle_max_deg": prepared["max_normal_angle_deg"],
                      "synthetic": prepared["synthetic"],
                      "executable_candidate": prepared["executable_candidate"],
                      "target_N": TARGET_FORCE, "wipe_mm_s": 3, "approach_mm_s": 1,
                      "max_approach_mm": validate_approach_limit(max_approach_m)*1000,
                      "max_surface_height_mm": prepared["max_surface_height_mm"],
                      "max_normal_angle_deg": prepared["max_normal_angle_deg"],
                      "free_space_force_limit_N": FREE_SPACE_FORCE_LIMIT_N,
                      "normal_force_limit_N": NORMAL_FORCE_LIMIT_N,
                      "fixed_orientation": not prepared.get("spatial_orientation", False),
                      "surface_normal_aligned": prepared.get("spatial_orientation", False),
                      "normal_alignment": ("exact_tcp_z_to_interpolated_surface_normal"
                                           if prepared.get("spatial_orientation", False)
                                           else "capture_orientation"),
                      "normal_smoothing_max_deg": prepared.get("normal_smoothing_max_deg", 0.0),
                      "requires_live_pose_and_free_space_tare": True},
                     indent=2, allow_nan=False))


def execute(plan, *, segment=None, ip="172.16.0.2", log_path=None, max_approach_m=DEFAULT_MAX_APPROACH_M,
            max_surface_height_m=DEFAULT_MAX_SURFACE_HEIGHT_M,
            max_normal_angle_deg=DEFAULT_MAX_NORMAL_ANGLE_DEG, full_strip=False, speed_scale=1.0):
    """Explicit hardware operation. No recovery, load/EE changes, or gripper commands."""
    max_approach_m = validate_approach_limit(max_approach_m)
    if not np.isfinite(speed_scale) or speed_scale<=0:
        raise ValueError('speed scale must be finite and positive')
    log_path = Path(log_path or f"output/wipe_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000:06d}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = {"robot_ip": ip, "status": "starting", "error": None, "samples": [],
           "force_basis": "-delta sensor Fz; TCP/EE Z aligned to local surface normal; software median tare",
           "cleanup_errors": [], "max_approach_mm": max_approach_m*1000,
           "free_space_force_limit_N": FREE_SPACE_FORCE_LIMIT_N, "force_control_mode": "normal_force_admittance",
           "total_force_limit_N": TOTAL_FORCE_LIMIT_N, "normal_force_limit_N": NORMAL_FORCE_LIMIT_N,
           "max_tensile_normal_N": MAX_TENSILE_NORMAL_N,
           "speed_scale": speed_scale,
           "wipe_speed_mm_s": 3*speed_scale, "search_speed_mm_s": speed_scale}
    robot = None
    started = time.monotonic()
    try:
        if full_strip:
            from .run_cycle import validate_full_plan
            validate_full_plan(plan, max_surface_height_m, max_normal_angle_deg, max_approach_m)
            if segment is None:
                raise ExecutionError('full-strip execution requires an explicit lane; no longest-fragment fallback')
            log['execution_admission'] = 'complete_full_strip_validated'
        p = prepare_plan(plan, segment, require_real=not full_strip, max_surface_height_m=max_surface_height_m,
                         max_normal_angle_deg=max_normal_angle_deg)
        log["max_surface_height_mm"] = p["max_surface_height_mm"]
        log["max_normal_angle_deg"] = p["max_normal_angle_deg"]
        log["segment_index"] = p["index"]
        log["normal_alignment"] = ("exact_tcp_z_to_interpolated_surface_normal"
                                    if p.get("spatial_orientation") else "capture_orientation")
        log["normal_smoothing_max_deg"] = p.get("normal_smoothing_max_deg", 0.0)
        # Lazy imports ensure preflight and pure-function tests never connect.
        import franky
        from .serial_sensor import KunweiSensor

        robot = franky.Robot(ip, relative_dynamics_factor=0.05,
                             default_torque_threshold=20.0, default_force_threshold=20.0)
        state = robot.state
        check_robot_state(state, (franky.RobotMode.Idle.value,), log, "entry")
        pose = np.asarray(state.O_T_EE.matrix, float).copy()
        entry = validate_live_pose(p, pose, state.F_T_EE.matrix, max_approach_m)
        log['approach_distance_mm'] = float(np.linalg.norm(entry-pose[:3,3])*1000)
        R, n, offset = p["rotation"], p["normal"], p["tcp_offset"]
        with KunweiSensor() as sensor:
            # Force sensing is deliberately disabled during free-space motion.
            # The serial reader is opened here so it can be enabled at the
            # contact handoff, but no force sample is required before then.
            force_enabled, tare = False, None
            ready_pose = pose.copy()
            log["ready_T_base_ee"] = ready_pose.tolist()
            last_command = pose[:3, 3].copy()
            last_rotation = pose[:3, :3].copy()
            last_tick = time.monotonic()
            phase, distance = "to_standoff", 0.0

            def tick():
                nonlocal last_tick
                delay = PERIOD - (time.monotonic() - last_tick)
                if delay > 0:
                    time.sleep(delay)
                now = time.monotonic()
                dt = now - last_tick
                last_tick = now
                if not 0 < dt <= MAX_LOOP_DT:
                    raise ExecutionError(f"control loop stalled: dt={dt:.4f} s")
                if force_enabled:
                    sample = sensor.latest(0.05)
                    # Before contact confirmation, permit small tensile/noise
                    # readings while the search continues toward the visual
                    # surface. Once contact is established, retain the normal
                    # reaction-direction guard.
                    force, delta = check_wrench(
                        sample.raw_sensor_6, tare,
                        reject_negative=(phase != "slow_approach"))
                    sensor_age = sample.age_s()
                else:
                    sample, force, delta, sensor_age = None, 0.0, np.zeros(6), None
                state = robot.state
                current = np.asarray(state.O_T_EE.matrix, float)
                joints = np.asarray(state.q, float)
                # Conservative guard for the joint-2 boundary encountered in
                # this cell. This is not a replacement for whole-path IK.
                if not np.isfinite(joints).all() or joints[1] >= 1.72:
                    raise ExecutionError('joint 2 reached conservative 1.72 rad guard; stop before previous boundary')
                check_robot_state(state, (franky.RobotMode.Idle.value,
                                          franky.RobotMode.Move.value), log, phase)
                if not np.isfinite(current).all():
                    raise ExecutionError("nonfinite robot pose")
                orientation_error = rotation_angle_deg(current[:3, :3], last_rotation)
                if orientation_error >= (5.0 if p.get("spatial_orientation") else 1.0):
                    raise ExecutionError("orientation tracking error exceeds the configured limit")
                tracking = float(np.linalg.norm(current[:3, 3] - last_command))
                surface = surface_at(p, distance)
                measured_normal = normal_at(p, distance) if p.get("spatial_orientation") else n
                measured_h = float((current[:3, 3] + current[:3, :3] @ offset - surface) @ measured_normal)
                log["samples"].append({"t_s": now - started, "phase": phase, "dt_s": dt,
                                       "force_N": force, "delta_sensor_6": delta.tolist(),
                                       "sensor_age_s": sensor_age, "force_control_enabled": force_enabled,
                                       "distance_m": distance,
                                       "measured_height_m": measured_h, "tracking_error_m": tracking,
                                       "orientation_error_deg": orientation_error,
                                       "joint_q_rad": joints.tolist(),
                                       "control_command_success_rate": float(state.control_command_success_rate),
                                       "command_ee_m": last_command.tolist(),
                                       "measured_ee_m": current[:3, 3].tolist()})
                if tracking > 0.010:
                    raise ExecutionError("position tracking error exceeds 10 mm")
                if measured_h < -MAX_PENETRATION:
                    raise ExecutionError("measured TCP exceeds 12 mm visual penetration")
                return dt, force

            def send(position, rotation=None):
                nonlocal last_command, last_rotation
                position = np.asarray(position, float)
                rotation = last_rotation if rotation is None else np.asarray(rotation, float)
                surface = surface_at(p, distance)
                command_normal = normal_at(p, distance) if p.get("spatial_orientation") else n
                if not np.isfinite(position).all():
                    raise ExecutionError("nonfinite target")
                if (position + rotation @ offset - surface) @ command_normal < -MAX_PENETRATION:
                    raise ExecutionError("commanded TCP exceeds 12 mm visual penetration")
                target = np.eye(4)
                target[:3, :3], target[:3, 3] = rotation, position
                robot.move(franky.CartesianMotion(franky.Affine(target)), asynchronous=True)
                last_command = position.copy()
                last_rotation = rotation.copy()

            def line_to(target, speed, *, expect_free=False, target_rotation=None):
                nonlocal last_command, last_rotation, last_tick
                speed *= speed_scale
                origin = last_command.copy()
                origin_rotation = last_rotation.copy()
                target_rotation = origin_rotation if target_rotation is None else np.asarray(target_rotation, float)
                if expect_free:
                    # Free-space motion has no force-feedback requirement.  A
                    # CartesianWaypoint with minimum_time lets the Franka
                    # controller generate a bounded-acceleration trajectory
                    # with the requested segment speed. A one-shot pose alone
                    # would otherwise use the robot's default timing.
                    target_pose = np.eye(4)
                    target_pose[:3, :3], target_pose[:3, 3] = target_rotation, np.asarray(target, float)
                    duration = max(1.0, np.linalg.norm(np.asarray(target, float) - origin) /
                                   max(speed, 1e-6) + WIPE_RAMP_TIME_S)
                    waypoint = franky.CartesianWaypoint(
                        franky.CartesianState(franky.RobotPose(franky.Affine(target_pose))),
                        minimum_time=franky.Duration(int(duration * 1000)))
                    robot.move(franky.CartesianWaypointMotion([waypoint]), asynchronous=False)
                    state = robot.state
                    check_robot_state(state, (franky.RobotMode.Idle.value,), log, phase)
                    actual = np.asarray(state.O_T_EE.matrix, float)
                    if np.linalg.norm(actual[:3, 3] - target) > 0.010:
                        raise ExecutionError("position tracking error exceeds 10 mm after free-space motion")
                    last_command = np.asarray(target, float).copy()
                    last_rotation = target_rotation.copy()
                    # The synchronous waypoint consumed the entire segment
                    # outside the 100 Hz loop. Start the next control period
                    # from now instead of treating that travel time as a
                    # stalled force-control cycle.
                    last_tick = time.monotonic()
                    return
                slerp = Slerp([0, 1], Rotation.from_matrix([origin_rotation, target_rotation]))
                # Quintic interpolation has zero velocity/acceleration at both ends.
                duration = max(0.5, np.linalg.norm(target - origin) * 1.875 / speed)
                elapsed = 0.0
                while elapsed < duration:
                    dt, force = tick()
                    if expect_free and force > FREE_SPACE_FORCE_LIMIT_N:
                        raise ExecutionError("unexpected contact before slow approach")
                    elapsed += motion_step(dt)
                    u = min(elapsed / duration, 1.0)
                    blend = u**3 * (10 - 15*u + 6*u*u)
                    send(origin + blend * (target - origin), slerp([blend]).as_matrix()[0])
                # Hold the final free-space target briefly before the next phase.
                settled = 0.0
                while settled < 0.25:
                    dt, force = tick()
                    if expect_free and force > FREE_SPACE_FORCE_LIMIT_N:
                        raise ExecutionError("unexpected contact while settling")
                    settled += dt

            line_to(entry, 0.010, expect_free=True, target_rotation=rotation_at(p, 0.0))
            phase = "to_search_start"
            search_normal = normal_at(p, 0.0) if p.get("spatial_orientation") else n
            search_rotation = rotation_at(p, 0.0)
            line_to(ee_position(p["points"][0], SEARCH_START, search_rotation, offset, search_normal),
                    0.005, expect_free=True, target_rotation=search_rotation)
            # Enable the force loop only at the fixed handoff pose. Take a new
            # quiet baseline here, after all free-space travel is complete.
            raw_samples, frame = [], 0
            tare_started = time.monotonic()
            while time.monotonic() - tare_started < 1.0:
                sample = sensor.wait_next(frame, timeout_s=0.05, max_age_s=0.05)
                frame = sample.frame
                raw_samples.append(sample.raw_sensor_6)
            tare, tare_std = tare_from_samples(raw_samples)
            force_enabled = True
            # The one-second baseline is outside the 100 Hz control loop; reset
            # the loop clock so its acquisition time is not reported as a stall.
            last_tick = time.monotonic()
            log["tare_raw_sensor_6"] = tare.tolist()
            log["tare_force_std_N"] = tare_std.tolist()
            log["force_control_enabled_from_phase"] = "slow_approach"
            phase, height = "slow_approach", SEARCH_START
            contact_time = 0.0
            while True:
                dt, force = tick()
                contact_time = (contact_time + dt if force > CONTACT_DETECT_FORCE_N else 0.0)
                if contact_time >= CONTACT_CONFIRM_S:
                    # Preserve the last streamed position at the handoff.
                    height = float((last_command + last_rotation @ offset - p["points"][0]) @ search_normal)
                    break
                height -= 0.001 * speed_scale * motion_step(dt)
                if height < -MAX_PENETRATION:
                    raise ExecutionError("no contact within 12 mm below the visual surface")
                send(ee_position(p["points"][0], height, search_rotation, offset, search_normal), search_rotation)
            admittance = NormalAdmittance(-0.001 * speed_scale)
            phase, elapsed, stable = "establish_force", 0.0, 0.0
            while stable < 0.25:
                dt, force = tick()
                elapsed += dt
                if elapsed > 15:
                    raise ExecutionError(f"{TARGET_FORCE} N force did not settle within 15 s")
                stable = stable + dt if abs(force - TARGET_FORCE) < FORCE_SETTLE_TOLERANCE_N else 0.0
                height = admittance.step(height, force, dt)
                send(ee_position(p["points"][0], height, search_rotation, offset, search_normal), search_rotation)
            # Keep the same planned orientation law through reversal; only
            # path progress reverses. The pass is constant speed after a
            # jerk-limited 0.5 s ramp.
            log["target_force_N"] = TARGET_FORCE
            log["planned_one_way_length_m"] = p["length_m"]
            log["passes"] = 2
            nominal_speed = 0.003 * speed_scale
            duration = max(2.0 * WIPE_RAMP_TIME_S + 0.1,
                           p["length_m"] / nominal_speed + WIPE_RAMP_TIME_S)
            for reverse in (False, True):
                phase = "wipe_backward" if reverse else "wipe_forward"
                lost, elapsed, progress_time = 0.0, 0.0, 0.0
                while progress_time < duration:
                    dt, force = tick()
                    elapsed += dt
                    lost = lost + dt if force <= CONTACT_PRESENT_FORCE_N else 0.0
                    if lost > FORCE_LOSS_TIMEOUT_S:
                        raise ExecutionError("contact lost for >2 s")
                    if elapsed > duration * 3 + 10:
                        raise ExecutionError("wipe did not complete within its time limit")
                    height = admittance.step(height, force, dt)
                    # Brief force dips are handled by normal admittance; do not
                    # toggle tangential speed. Sustained loss still aborts above.
                    progress_time = min(duration, progress_time + motion_step(dt))
                    distance = pass_distance(progress_time, duration, p["length_m"], reverse)
                    path_rotation = rotation_at(p, distance)
                    path_normal = normal_at(p, distance) if p.get("spatial_orientation") else n
                    send(ee_position(surface_at(p, distance), height, path_rotation, offset, path_normal), path_rotation)
            phase = "retreat"
            end_normal = normal_at(p, distance) if p.get("spatial_orientation") else n
            line_to(last_command + STANDOFF * end_normal, 0.005, expect_free=True,
                    target_rotation=rotation_at(p, distance))
            # The TCP is now clear of the surface; no force sample is needed
            # for the remaining free-space return.
            force_enabled = False
            # Retrace the validated free-space entry after unloading contact.
            # This is the start pose of THIS run, not a saved historical pose.
            phase = "return_to_precontact"
            line_to(entry, 0.005, expect_free=True, target_rotation=rotation_at(p, 0.0))
            phase = "return_to_ready"
            line_to(ready_pose[:3, 3], 0.010, expect_free=True, target_rotation=ready_pose[:3, :3])
            settle_started = time.monotonic()
            while True:
                _, force = tick()
                if abs(force) > FREE_SPACE_FORCE_LIMIT_N:
                    raise ExecutionError("unexpected contact while returning to ready pose")
                reached = np.asarray(robot.state.O_T_EE.matrix, float)
                if (np.linalg.norm(reached[:3, 3] - ready_pose[:3, 3]) < 0.001 and
                        rotation_angle_deg(reached[:3, :3], ready_pose[:3, :3]) < 0.5):
                    break
                if time.monotonic() - settle_started > 3:
                    raise ExecutionError("return to ready pose did not settle")
            log["returned_to_ready"] = True
            log["status"] = "completed"
    except BaseException as exc:
        log["status"], log["error"] = "aborted", f"{type(exc).__name__}: {exc}"
        # Capture before stop/join can overwrite the original controller fault.
        if robot is not None and "robot_fault" not in log:
            try:
                check_robot_state(robot.state, (franky.RobotMode.Idle.value,
                                                franky.RobotMode.Move.value), log,
                                  locals().get("phase"))
            except Exception as diagnostic_error:
                log["fault_capture_message"] = str(diagnostic_error)
        raise
    finally:
        # A fault does not trigger a blind retreat or automatic error recovery.
        if robot is not None:
            for name, args in (("stop", ()), ("join_motion", (5.0,))):
                try:
                    getattr(robot, name)(*args)
                except Exception as exc:
                    log["cleanup_errors"].append(f"{name}: {exc}")
        log["duration_s"] = time.monotonic() - started
        log_path.write_text(json.dumps(log, indent=2, allow_nan=False) + "\n")
        print(f"Execution log: {log_path}")
    return log_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--segment", type=int, help="zero-based index; default: longest valid <=200 mm segment")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--max-surface-height-mm", type=float, default=DEFAULT_MAX_SURFACE_HEIGHT_M*1000)
    parser.add_argument("--max-normal-angle-deg", type=float, default=DEFAULT_MAX_NORMAL_ANGLE_DEG)
    parser.add_argument("--max-approach-mm", type=float, default=DEFAULT_MAX_APPROACH_M*1000,
                        help="maximum free-space travel to precontact in mm (default: 500; controlled obstacle-free experiment)")
    parser.add_argument("--execute", action="store_true", help="connect to hardware and move the robot")
    args = parser.parse_args(argv)
    try:
        max_approach_m = validate_approach_limit(args.max_approach_mm/1000)
        plan = json.loads(args.plan.read_text())
        if args.execute:
            execute(plan, segment=args.segment, ip=args.ip, log_path=args.log, max_approach_m=max_approach_m,
                    max_surface_height_m=args.max_surface_height_mm/1000, max_normal_angle_deg=args.max_normal_angle_deg)
        else:
            print_preflight(prepare_plan(plan, args.segment, max_surface_height_m=args.max_surface_height_mm/1000,
                                         max_normal_angle_deg=args.max_normal_angle_deg), max_approach_m)
    except (ExecutionError, OSError, ValueError, KeyError) as exc:
        print(f"Stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
