"""Conservative, offline red-mark paths on registered RGB optical-frame points.

Each path stays on one contiguous, observed red image-row fragment. Local PCA
uses surrounding valid ROI points, never a global plane or filled-in depth.
The result is a preview: reachability, collision and eraser footprint are not
checked here, and the row spacing does not guarantee complete wipe coverage.
"""

from __future__ import annotations

from collections import Counter
import math

import cv2
import numpy as np


DEFAULT_CONFIG = {
    "sample_step_m": 0.004,
    "row_spacing_m": 0.010,
    "standoff_m": 0.050,
    "neighborhood_radius_px": 5,
    "min_neighbors": 18,
    "max_neighbor_distance_m": 0.025,
    "max_depth_jump_m": 0.012,
    "max_surface_residual_m": 0.003,
    "max_normal_angle_deg": 25.0,
    "max_scan_direction_angle_deg": 45.0,
    "max_view_angle_deg": 75.0,
    "min_segment_points": 2,
    "min_tangent_eigenvalue_m2": 1e-7,
    "min_tangent_eigenvalue_ratio": 0.03,
    "min_component_pixels": 1,
    "hsv_low_1": [0, 80, 50],
    "hsv_high_1": [12, 255, 255],
    "hsv_low_2": [168, 80, 50],
    "hsv_high_2": [179, 255, 255],
}

# Target names accepted by the full-strip entry point.  The detector returns
# the same binary mask for every color, so the geometry and force-control
# stages do not depend on which color was selected.
TARGET_COLORS = (
    'red', 'orange', 'yellow', 'lime', 'green', 'cyan', 'blue',
    'violet', 'purple', 'magenta', 'pink', 'brown', 'gray', 'white', 'black',
)

# OpenCV hue is 0..179.  Saturation/value limits intentionally leave some
# margin for camera exposure changes; component selection below rejects large
# background regions and keeps one elongated wiping object.
HSV_TARGET_RANGES = {
    'orange': ((5, 80, 45), (24, 255, 255)),
    'yellow': ((20, 70, 45), (40, 255, 255)),
    'lime': ((35, 70, 35), (65, 255, 255)),
    'green': ((40, 70, 30), (90, 255, 255)),
    'cyan': ((80, 60, 35), (105, 255, 255)),
    'blue': ((95, 60, 30), (135, 255, 255)),
    'violet': ((125, 55, 30), (150, 255, 255)),
    'purple': ((130, 55, 30), (165, 255, 255)),
    'magenta': ((145, 55, 35), (179, 255, 255)),
    'pink': ((155, 35, 70), (179, 255, 255)),
    'brown': ((5, 70, 25), (25, 255, 190)),
    'gray': ((0, 0, 35), (179, 55, 210)),
}


def segment_red(
    bgr, roi, hsv_low_1=(0, 80, 50), hsv_high_1=(12, 255, 255),
    hsv_low_2=(168, 80, 50), hsv_high_2=(179, 255, 255),
    min_component_pixels=1,
):
    """Return a bool mask. ROI is mandatory; no morphology fills gaps."""
    bgr = np.asarray(bgr)
    roi = np.asarray(roi)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr must be H x W x 3 uint8")
    if roi.dtype != np.bool_ or roi.shape != bgr.shape[:2]:
        raise ValueError("roi must be an explicit H x W boolean mask")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, np.asarray(hsv_low_1), np.asarray(hsv_high_1)) |
            cv2.inRange(hsv, np.asarray(hsv_low_2), np.asarray(hsv_high_2))) > 0
    mask &= roi
    if min_component_pixels > 1:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component_pixels
        mask = keep[labels]
    return mask


def segment_white(bgr, roi, *, max_saturation=80, min_value=170,
                  min_component_pixels=1, min_aspect_ratio=3.0):
    """Return the longest bright, low-saturation elongated component.

    White tape is commonly merged with the light workpiece, so unlike the
    red detector this mode selects an elongated connected component rather
    than requiring the entire ROI to contain only one bright component.
    """
    bgr = np.asarray(bgr)
    roi = np.asarray(roi)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr must be H x W x 3 uint8")
    if roi.dtype != np.bool_ or roi.shape != bgr.shape[:2]:
        raise ValueError("roi must be an explicit H x W boolean mask")
    if (not np.isfinite([max_saturation, min_value, min_component_pixels, min_aspect_ratio]).all()
            or not 0 <= max_saturation <= 255 or not 0 <= min_value <= 255
            or min_component_pixels < 1 or min_aspect_ratio <= 1):
        raise ValueError("invalid white segmentation parameters")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] <= int(max_saturation)) & (hsv[..., 2] >= int(min_value)) & roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if area < min_component_pixels or width < min_aspect_ratio * max(height, 1):
            continue
        candidates.append((int(area), i))
    if not candidates:
        return np.zeros_like(roi)
    _, selected = max(candidates)
    return labels == selected


def segment_black(bgr, roi, *, max_value=90, min_component_pixels=80,
                  min_aspect_ratio=3.0):
    """Return the largest non-border dark elongated component."""
    bgr = np.asarray(bgr)
    roi = np.asarray(roi)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr must be H x W x 3 uint8")
    if roi.dtype != np.bool_ or roi.shape != bgr.shape[:2]:
        raise ValueError("roi must be an explicit H x W boolean mask")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 2] <= int(max_value)) & roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        touches_border = x == 0 or y == 0 or x + width >= mask.shape[1] or y + height >= mask.shape[0]
        if area < min_component_pixels or touches_border:
            continue
        yy, xx = np.where(labels == i)
        if len(xx) < 3:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(np.column_stack((xx, yy)).T))
        elongation = np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-12))
        if elongation < min_aspect_ratio:
            continue
        candidates.append((int(area), i))
    if not candidates:
        return np.zeros_like(roi)
    _, selected = max(candidates)
    return labels == selected


def segment_hsv_component(bgr, roi, hsv_low, hsv_high, *,
                          min_component_pixels=80, min_aspect_ratio=2.5):
    """Select one elongated, non-border component from an HSV range."""
    bgr = np.asarray(bgr)
    roi = np.asarray(roi)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("bgr must be H x W x 3 uint8")
    if roi.dtype != np.bool_ or roi.shape != bgr.shape[:2]:
        raise ValueError("roi must be an explicit H x W boolean mask")
    low, high = np.asarray(hsv_low, dtype=np.uint8), np.asarray(hsv_high, dtype=np.uint8)
    if low.shape != (3,) or high.shape != (3,) or np.any(low > high):
        raise ValueError("HSV bounds must be ordered 3-vectors")
    if min_component_pixels < 1 or min_aspect_ratio <= 1:
        raise ValueError("invalid component selection parameters")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, low, high) > 0) & roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    candidates = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        touches_border = (x == 0 or y == 0 or
                          x + width >= mask.shape[1] or y + height >= mask.shape[0])
        if area < min_component_pixels or touches_border:
            continue
        yy, xx = np.where(labels == i)
        if len(xx) < 3:
            continue
        eigenvalues = np.linalg.eigvalsh(np.cov(np.column_stack((xx, yy)).T))
        elongation = np.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-12))
        if elongation >= min_aspect_ratio:
            candidates.append((int(area), i))
    if not candidates:
        return np.zeros_like(roi)
    _, selected = max(candidates)
    return labels == selected


def segment_target(bgr, roi, target_color='red'):
    """Segment the configured wiping target while preserving red defaults."""
    if target_color == 'red':
        return segment_red(bgr, roi)
    if target_color == 'white':
        return segment_white(bgr, roi, max_saturation=100, min_value=200,
                             min_component_pixels=80, min_aspect_ratio=3.0)
    if target_color == 'black':
        return segment_black(bgr, roi, max_value=90,
                             min_component_pixels=80, min_aspect_ratio=3.0)
    if target_color in HSV_TARGET_RANGES:
        low, high = HSV_TARGET_RANGES[target_color]
        return segment_hsv_component(bgr, roi, low, high,
                                     min_component_pixels=80,
                                     min_aspect_ratio=2.5)
    raise ValueError(f"target_color must be one of {', '.join(TARGET_COLORS)}")


def _transform(value, name):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all() or
            not np.allclose(value[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"{name} must be a finite rigid 4 x 4 transform")
    return value


def _unit(value):
    length = np.linalg.norm(value)
    if length < 1e-12:
        raise ValueError("degenerate direction")
    return value / length


def _local_geometry(xyz, valid, u, v, cfg):
    """Fit a local plane for a normal, while preserving the measured point."""
    p = xyz[v, u]
    radius = cfg["neighborhood_radius_px"]
    y0, y1 = max(0, v - radius), min(xyz.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(xyz.shape[1], u + radius + 1)
    vv, uu = np.mgrid[y0:y1, x0:x1]
    patch = xyz[y0:y1, x0:x1]
    usable = valid[y0:y1, x0:x1].copy()
    usable &= np.linalg.norm(patch - p, axis=-1) <= cfg["max_neighbor_distance_m"]
    points = patch[usable]
    if len(points) < cfg["min_neighbors"]:
        return None, "small_neighborhood"
    center = points.mean(axis=0)
    delta = points - center
    eigenvalues, eigenvectors = np.linalg.eigh(delta.T @ delta / len(points))
    if (eigenvalues[1] < cfg["min_tangent_eigenvalue_m2"] or
            eigenvalues[1] / max(eigenvalues[2], 1e-20) < cfg["min_tangent_eigenvalue_ratio"]):
        return None, "degenerate_neighborhood"
    normal = eigenvectors[:, 0]
    residual = np.abs(delta @ normal)
    # RMS and a center check reject noisy patches and isolated wrong depths.
    if (np.sqrt(np.mean(residual ** 2)) > cfg["max_surface_residual_m"] or
            abs((p - center) @ normal) > cfg["max_surface_residual_m"]):
        return None, "surface_residual"
    view = _unit(-p)
    if normal @ view < 0:
        normal = -normal
    if normal @ view < math.cos(math.radians(cfg["max_view_angle_deg"])):
        return None, "view_angle"
    # Regress against +u, independent of snake traversal order. This prevents
    # flipping the tool 180 degrees when reversing a raster row.
    design = np.column_stack((uu[usable] - u, vv[usable] - v, np.ones(len(points))))
    derivatives, _, rank, _ = np.linalg.lstsq(design, points - p, rcond=None)
    tangent = derivatives[0] - normal * (derivatives[0] @ normal)
    if rank < 3 or np.linalg.norm(tangent) < 1e-10:
        return None, "degenerate_scan_direction"
    return {"pixel": np.array([float(u), float(v)]), "point": p,
            "normal": normal, "tangent": _unit(tangent)}, None


def _choose_rows(mask, xyz, valid, cfg):
    """Bound stride by the largest observed local physical row increment."""
    active = np.flatnonzero(mask.any(axis=1))
    pairs = valid[1:] & valid[:-1]
    distances = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    increments = distances[pairs & (distances > 1e-9) &
                           (distances <= cfg["max_depth_jump_m"])]
    max_increment = float(increments.max()) if increments.size else None
    stride = max(1, int(cfg["row_spacing_m"] / max_increment)) if max_increment else 1
    # Keep both boundaries of each active row run. Non-target rows never turn
    # into paths, even if they were used as neighbors for estimating normals.
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    rows = []
    for group in groups:
        if len(group):
            rows.extend(group[::stride].tolist())
            rows.append(int(group[-1]))
    return sorted(set(rows)), stride, max_increment


def _resample(fragment, step):
    """Interpolate only along pairs of adjacent, accepted red pixels."""
    points = np.asarray([item["point"] for item in fragment])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    if arc[-1] < 1e-10:
        return []
    distances = np.r_[np.arange(0.0, arc[-1], step), arc[-1]]
    result = []
    for distance in distances:
        index = min(int(np.searchsorted(arc, distance, side="right")) - 1, len(fragment) - 2)
        while lengths[index] < 1e-12 and index < len(lengths) - 1:
            index += 1
        fraction = np.clip((distance - arc[index]) / max(lengths[index], 1e-12), 0.0, 1.0)
        a, b = fragment[index], fragment[index + 1]
        item = {key: a[key] * (1 - fraction) + b[key] * fraction for key in a}
        item["normal"] = _unit(item["normal"])
        item["tangent"] = _unit(item["tangent"])
        result.append(item)
    return result


def plan_surface(bgr, xyz_rgb_m, roi, T_base_camera, T_ee_tcp, config=None):
    """Build JSON-serializable contact/standoff paths; does not move hardware.

    xyz_rgb_m must already correspond to BGR pixels and be expressed in the
    RGB optical frame (metres). Normals face the camera. TCP +Z points into the
    surface, +X follows increasing image u. EE = TCP @ inverse(EE->TCP).
    """
    cfg = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown geometry config: {sorted(unknown)}")
        cfg.update(config)
    for key in ("sample_step_m", "row_spacing_m", "max_neighbor_distance_m",
                "max_depth_jump_m", "max_surface_residual_m", "min_tangent_eigenvalue_m2"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("neighborhood_radius_px", "min_neighbors", "min_segment_points", "min_component_pixels"):
        if not isinstance(cfg[key], (int, np.integer)) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["min_neighbors"] < 3 or cfg["min_segment_points"] < 2:
        raise ValueError("min_neighbors >= 3 and min_segment_points >= 2 are required")
    if not np.isfinite(cfg["standoff_m"]) or cfg["standoff_m"] < 0:
        raise ValueError("standoff_m must be finite and nonnegative")
    for key in ("max_normal_angle_deg", "max_scan_direction_angle_deg", "max_view_angle_deg"):
        if not np.isfinite(cfg[key]) or not 0 < cfg[key] < 90:
            raise ValueError(f"{key} must be between 0 and 90 degrees")
    if not 0 < cfg["min_tangent_eigenvalue_ratio"] <= 1:
        raise ValueError("min_tangent_eigenvalue_ratio must be in (0, 1]")
    mask = segment_red(bgr, roi, **{key: cfg[key] for key in
                       ("hsv_low_1", "hsv_high_1", "hsv_low_2", "hsv_high_2", "min_component_pixels")})
    xyz = np.asarray(xyz_rgb_m, dtype=float)
    if xyz.shape != (*mask.shape, 3):
        raise ValueError("xyz_rgb_m must have shape H x W x 3 matching BGR")
    T_bc = _transform(T_base_camera, "T_base_camera")
    T_et = _transform(T_ee_tcp, "T_ee_tcp")
    T_te = np.linalg.inv(T_et)
    valid = np.asarray(roi) & np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    target = mask & valid
    if not target.any():
        raise ValueError("no red target with valid depth inside explicit ROI")
    rows, stride, increment = _choose_rows(target, xyz, valid, cfg)
    segments, rejected = [], Counter()
    cos_normal = math.cos(math.radians(cfg["max_normal_angle_deg"]))
    cos_tangent = math.cos(math.radians(cfg["max_scan_direction_angle_deg"]))

    def append_fragment(fragment, row):
        if len(fragment) < cfg["min_segment_points"]:
            if fragment:
                rejected["short_fragment"] += 1
            return
        samples = _resample(fragment, cfg["sample_step_m"])
        if len(samples) < 2:
            return
        if len(segments) % 2:
            samples.reverse()
        waypoints = []
        for sample in samples:
            normal = T_bc[:3, :3] @ sample["normal"]
            point = T_bc[:3, :3] @ sample["point"] + T_bc[:3, 3]
            z = -normal
            x = T_bc[:3, :3] @ sample["tangent"]
            x = _unit(x - z * (x @ z))
            y = _unit(np.cross(z, x))
            x = _unit(np.cross(y, z))
            contact = np.eye(4)
            contact[:3, :3] = np.column_stack((x, y, z))
            contact[:3, 3] = point
            standoff = contact.copy()
            standoff[:3, 3] += normal * cfg["standoff_m"]
            waypoints.append({
                "pixel_uv": sample["pixel"].tolist(),
                "surface_point_base_m": point.tolist(),
                "normal_out_base": normal.tolist(),
                "T_base_tcp_contact": contact.tolist(),
                "T_base_ee_contact": (contact @ T_te).tolist(),
                "T_base_tcp_standoff": standoff.tolist(),
                "T_base_ee_standoff": (standoff @ T_te).tolist(),
            })
        segments.append({"row_v": int(row), "waypoints": waypoints,
                         "length_m": float(sum(np.linalg.norm(b["point"] - a["point"])
                                               for a, b in zip(fragment, fragment[1:])))})

    for v in rows:
        fragment = []
        for u in range(mask.shape[1]):
            if not target[v, u]:
                append_fragment(fragment, v)
                fragment = []
                continue
            item, reason = _local_geometry(xyz, valid, u, v, cfg)
            if reason:
                rejected[reason] += 1
                append_fragment(fragment, v)
                fragment = []
                continue
            if fragment:
                previous = fragment[-1]
                discontinuity = (
                    np.linalg.norm(item["point"] - previous["point"]) > cfg["max_depth_jump_m"] or
                    item["normal"] @ previous["normal"] < cos_normal or
                    item["tangent"] @ previous["tangent"] < cos_tangent)
                if discontinuity:
                    rejected["discontinuity_split"] += 1
                    append_fragment(fragment, v)
                    fragment = []
            fragment.append(item)
        append_fragment(fragment, v)
    if not segments:
        raise ValueError(f"no acceptable red surface segment; rejected={dict(rejected)}")
    return {"segments": segments, "metadata": {
        "offline_preview_only": True,
        "input_point_frame": "rgb_optical",
        "normal_convention": "outward toward camera; TCP +Z = -normal",
        "surface_method": "local ROI neighborhood PCA at measured red pixels",
        "sampling_method": "3D arc length within adjacent valid red pixels",
        "coverage_guaranteed": False,
        "reachability_checked": False,
        "collision_checked": False,
        "eraser_footprint_checked": False,
        "between_segments_motion_planned": False,
        "red_pixels": int(mask.sum()), "red_valid_depth_pixels": int(target.sum()),
        "selected_rows": rows, "row_stride_pixels": stride,
        "max_observed_row_increment_m": increment,
        "rejected": dict(rejected), "config": cfg,
    }}
