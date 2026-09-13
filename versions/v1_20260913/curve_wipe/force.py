"""Six-axis static compensation and SIMULATION-ONLY normal admittance.

Transforms T_a_b map coordinates in b into a. External wrench means the
ENVIRONMENT acting ON THE TOOL. Its compressive normal component is positive
along the surface outward normal. No hardware APIs or calibration files are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array


def _transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError(f"{name} must be a rigid transform with proper rotation")
    return matrix


def _normal(value):
    vector = _vector(value, 3, "outward_normal_base")
    length = np.linalg.norm(vector)
    if not np.isclose(length, 1.0, atol=1e-3):
        raise ValueError("outward_normal_base must be a unit vector")
    return vector / length


@dataclass(frozen=True)
class CompensatedWrench:
    force_base_N: np.ndarray
    torque_at_tcp_base_Nm: np.ndarray
    external_sensor_6: np.ndarray

    @property
    def wrench_base_at_tcp_6(self):
        return np.concatenate((self.force_base_N, self.torque_at_tcp_base_Nm))


def compensate_wrench(raw_sensor_6, *, T_base_ee, T_ee_sensor, T_ee_tcp,
                      bias_sensor_6, gravity_base_N, com_sensor_m,
                      sensor_sign):
    """Remove static tool gravity and raw-channel bias; shift moment to TCP.

    All parameters are explicit: NO old mount/bias is assumed. sensor_sign must
    be +1 or -1, confirmed for the current mounting, and multiplies all six
    channels after raw bias removal. The measurement convention is:
        raw = bias + sensor_sign * (gravity_wrench + external_wrench).
    gravity_base_N is the tool's physical weight VECTOR (e.g. [0,0,-m*g]);
    com_sensor_m is sensor origin -> tool centre of mass, in sensor axes.
    Inertial forces and temperature drift are not compensated by this model.
    """
    raw = _vector(raw_sensor_6, 6, "raw_sensor_6")
    bias = _vector(bias_sensor_6, 6, "bias_sensor_6")
    gravity = _vector(gravity_base_N, 3, "gravity_base_N")
    com = _vector(com_sensor_m, 3, "com_sensor_m")
    base_ee = _transform(T_base_ee, "T_base_ee")
    ee_sensor = _transform(T_ee_sensor, "T_ee_sensor")
    ee_tcp = _transform(T_ee_tcp, "T_ee_tcp")
    if sensor_sign not in (-1, 1):
        raise ValueError("sensor_sign must be explicitly +1 or -1")
    base_sensor, base_tcp = base_ee @ ee_sensor, base_ee @ ee_tcp
    rotation = base_sensor[:3, :3]
    gravity_sensor = rotation.T @ gravity
    gravity_wrench = np.concatenate((gravity_sensor, np.cross(com, gravity_sensor)))
    external_sensor = sensor_sign * (raw - bias) - gravity_wrench
    force_base = rotation @ external_sensor[:3]
    # Moment about new origin q: M_q = M_p + (p - q) cross F.
    torque_tcp = (rotation @ external_sensor[3:]
                  + np.cross(base_sensor[:3, 3] - base_tcp[:3, 3], force_base))
    return CompensatedWrench(force_base, torque_tcp, external_sensor)


def normal_force(wrench, outward_normal_base):
    """Fn > 0 is compressive environmental reaction on the eraser."""
    force = (wrench.force_base_N if isinstance(wrench, CompensatedWrench)
             else _vector(wrench, 3, "force_base_N"))
    return float(_normal(outward_normal_base) @ force)


class ForceSafetyStop(RuntimeError):
    """Latched simulation stop; it must never be treated as a zero-force sample."""


@dataclass(frozen=True)
class NormalAdmittanceConfig:
    """Illustrative SIMULATION parameters, not a validated robot configuration."""
    target_force_N: float = 1.0
    virtual_mass_kg: float = 0.2
    damping_Ns_m: float = 80.0
    max_velocity_m_s: float = 0.003
    max_displacement_m: float = 0.008
    max_force_N: float = 8.0
    max_torque_Nm: float = 0.4
    max_tensile_normal_N: float = 0.5
    max_sample_age_s: float = 0.05
    max_dt_s: float = 0.02

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.target_force_N >= self.max_force_N:
            raise ValueError("target force must be below the force stop limit")


@dataclass(frozen=True)
class NormalStep:
    normal_force_N: float
    force_error_N: float
    offset_m: float
    velocity_m_s: float
    delta_base_m: np.ndarray


class NormalAdmittance:
    """Pure numerical law M*dv/dt + B*v = Fn - target.

    Positive offset/velocity points outward. Insufficient force therefore
    produces negative velocity, into the surface. Speed is clamped; reaching
    the travel bound STOPS and latches an exception, including in zero force.
    Input force/moment limits are checked before any filtering or motion.
    This is not a robot controller and does not implement collision avoidance.
    """
    def __init__(self, config=None):
        self.config = config or NormalAdmittanceConfig()
        self.offset_m = 0.0
        self.velocity_m_s = 0.0
        self.stop_reason = None

    def _stop(self, reason):
        self.velocity_m_s = 0.0
        self.stop_reason = reason
        raise ForceSafetyStop(reason)

    def step(self, force_base_N, torque_at_tcp_base_Nm, outward_normal_base,
             *, dt_s, sample_age_s):
        if self.stop_reason is not None:
            raise ForceSafetyStop(f"latched stop: {self.stop_reason}")
        try:
            force = _vector(force_base_N, 3, "force_base_N")
            moment = _vector(torque_at_tcp_base_Nm, 3, "torque_at_tcp_base_Nm")
            normal = _normal(outward_normal_base)
        except ValueError as exc:
            self._stop(str(exc))
        cfg = self.config
        if not math.isfinite(dt_s) or dt_s <= 0 or dt_s > cfg.max_dt_s:
            self._stop("invalid or excessive dt_s")
        if (not math.isfinite(sample_age_s) or sample_age_s < 0
                or sample_age_s > cfg.max_sample_age_s):
            self._stop("stale or invalid force sample")
        if np.linalg.norm(force) >= cfg.max_force_N:
            self._stop("force norm limit exceeded")
        if np.linalg.norm(moment) >= cfg.max_torque_Nm:
            self._stop("TCP torque norm limit exceeded")
        fn = float(force @ normal)
        if fn < -cfg.max_tensile_normal_N:
            self._stop("negative normal force: check contact direction and sensor sign")
        desired_v = (fn - cfg.target_force_N) / cfg.damping_Ns_m
        alpha = math.exp(-cfg.damping_Ns_m * dt_s / cfg.virtual_mass_kg)
        velocity = alpha * self.velocity_m_s + (1 - alpha) * desired_v
        velocity = float(np.clip(velocity, -cfg.max_velocity_m_s, cfg.max_velocity_m_s))
        delta = velocity * dt_s
        proposed_offset = self.offset_m + delta
        if abs(proposed_offset) >= cfg.max_displacement_m:
            self._stop("normal displacement limit reached (contact absent or force unachievable)")
        self.offset_m, self.velocity_m_s = proposed_offset, velocity
        return NormalStep(fn, cfg.target_force_N - fn, self.offset_m,
                          velocity, delta * normal)


def simulate_normal_force(*, config=None, duration_s=6.0, dt_s=0.005,
                          spring_stiffness_N_m=800.0, initial_gap_m=0.001):
    """Analytic unilateral spring demo; return JSON-ready report and trajectory.

    Surface reaction is Fn=k*max(0,-(gap+offset)); no robot or sensor is opened.
    A zero-stiffness/no-contact case must stop at the travel bound, not silently
    declare success. The reported final force is evaluated at the final offset.
    """
    for name, value in (("duration_s", duration_s), ("dt_s", dt_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(spring_stiffness_N_m) or spring_stiffness_N_m < 0:
        raise ValueError("spring_stiffness_N_m must be finite and nonnegative")
    if not math.isfinite(initial_gap_m) or initial_gap_m < 0:
        raise ValueError("initial_gap_m must be finite and nonnegative")
    core = NormalAdmittance(config)
    rows, status, reason = [], "completed", None
    count = int(math.ceil(duration_s / dt_s))
    for index in range(count):
        elapsed = index * dt_s
        current_dt = min(dt_s, duration_s - elapsed)
        force = spring_stiffness_N_m * max(0.0, -(initial_gap_m + core.offset_m))
        try:
            step = core.step([0, 0, force], [0, 0, 0], [0, 0, 1],
                             dt_s=current_dt, sample_age_s=0.0)
        except ForceSafetyStop as exc:
            status, reason = "stopped", str(exc)
            break
        rows.append({"time_s": elapsed + current_dt, "measured_force_N": force,
                     "commanded_offset_m": step.offset_m,
                     "normal_velocity_m_s": step.velocity_m_s})
    final_force = spring_stiffness_N_m * max(0.0, -(initial_gap_m + core.offset_m))
    error = core.config.target_force_N - final_force
    report = {"mode": "simulation_only", "status": status, "stop_reason": reason,
              "converged": status == "completed" and abs(error) <= 0.05,
              "target_force_N": core.config.target_force_N,
              "final_force_N": final_force, "final_error_N": error,
              "final_offset_m": core.offset_m,
              "max_abs_velocity_m_s": max((abs(x["normal_velocity_m_s"]) for x in rows), default=0),
              "steps": len(rows), "config": asdict(core.config),
              "model": {"spring_stiffness_N_m": spring_stiffness_N_m,
                        "initial_gap_m": initial_gap_m}}
    return {"report": report, "trajectory": rows}
