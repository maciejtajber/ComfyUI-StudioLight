from __future__ import annotations

"""
ComfyUI-StudioLight — Metric Camera-Space 3D Shadow Raymarching Engine
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Replaces 2D screen-space pixel stepping and arbitrary 40-pixel ceilings with
true 3D metric camera-space raymarching through the reconstructed depth field.
"""

from dataclasses import dataclass
import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from .color_utils import MODIFIER_PRESETS

_EPS = 1e-6


@dataclass(frozen=True)
class CameraModel:
    """Perspective camera intrinsics matching the StudioLight back-projection model."""

    vertical_fov_deg: float = 40.0
    aspect: float = 1.0
    image_y_down: bool = True
    near_m: float = 0.05
    far_m: float = 100.0

    @property
    def tan_half_fov(self) -> float:
        return math.tan(math.radians(self.vertical_fov_deg) * 0.5)


@dataclass(frozen=True)
class EmitterSpec:
    """2D metric aperture geometry for photographic studio modifiers."""

    shape: Literal["rectangle", "circle", "octagon"]
    width_m: float = 0.30
    height_m: float = 0.30
    radius_m: float = 0.15
    inner_radius_m: float = 0.0
    one_sided: bool = True


def _build_modifier_emitters() -> Dict[str, EmitterSpec]:
    """Derives metric 3D emitter specifications directly from MODIFIER_PRESETS (Single Source of Truth)."""
    emitters: Dict[str, EmitterSpec] = {}
    for name, spec in MODIFIER_PRESETS.items():
        geom = spec.get("geometry", "circle")
        if geom == "octagon":
            d = 2.0 * float(spec["radius_m"])
            emitters[name] = EmitterSpec("octagon", width_m=d, height_m=d, radius_m=float(spec["radius_m"]))
        elif geom == "rectangle":
            emitters[name] = EmitterSpec(
                "rectangle",
                width_m=float(spec["width_m"]),
                height_m=float(spec["height_m"]),
            )
        else:
            r_out = float(spec["radius_m"])
            r_in = r_out * float(spec.get("deflector_ratio", 0.0))
            emitters[name] = EmitterSpec("circle", radius_m=r_out, inner_radius_m=r_in)
    emitters["Direct Flash"] = emitters["Direct On-Camera Flash"]
    return emitters


MODIFIER_EMITTERS: Dict[str, EmitterSpec] = _build_modifier_emitters()


def _ensure_bhw(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 3:
        raise ValueError(f"{name} must be [B,H,W] or [B,H,W,1], got {tuple(x.shape)}")
    return x


def _ensure_bhw3(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"{name} must be [B,H,W,3], got {tuple(x.shape)}")
    return x


def project_camera_points(
    points: torch.Tensor,
    camera: CameraModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projects camera-space 3D points into grid_sample normalized coordinates [-1, 1]."""
    if points.shape[-1] != 3:
        raise ValueError("points last dimension must be 3")

    z = points[..., 2]
    cam_depth = -z
    tan_half = points.new_tensor(camera.tan_half_fov)
    aspect = points.new_tensor(float(camera.aspect))

    x_ndc = points[..., 0] / (torch.clamp(cam_depth, min=_EPS) * tan_half * aspect)
    y_ndc = points[..., 1] / (torch.clamp(cam_depth, min=_EPS) * tan_half)

    y_grid = -y_ndc if camera.image_y_down else y_ndc

    grid = torch.stack((x_ndc, y_grid), dim=-1)
    valid = (
        (cam_depth > camera.near_m)
        & (cam_depth < camera.far_m)
        & (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )
    return grid, cam_depth, valid


def _sample_depth(
    depth_bhw: torch.Tensor,
    grid_bhw2: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Bilinear depth lookup using the camera projection grid."""
    depth_b1hw = depth_bhw[:, None]
    sampled = F.grid_sample(
        depth_b1hw,
        grid_bhw2,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    return torch.where(valid, sampled, torch.full_like(sampled, float("nan")))


def _hammersley_2d(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic low-discrepancy 2D samples in [0,1)^2 (MPS-safe)."""
    if n <= 0:
        raise ValueError("n must be > 0")

    i = torch.arange(n, dtype=torch.int64)
    bits = i.clone()
    radical = torch.zeros(n, dtype=torch.float32)
    inv = 0.5
    while torch.any(bits > 0):
        radical = radical + (bits & 1).to(torch.float32) * inv
        bits = bits >> 1
        inv *= 0.5
    u = (i.to(torch.float32) + 0.5) / n
    samples = torch.stack((u, radical), dim=-1)
    return samples.to(device=device, dtype=dtype)


def _regular_polygon_vertices(n: int, radius_x: float, radius_y: float, device, dtype) -> torch.Tensor:
    theta = torch.arange(n, device=device, dtype=dtype) * (2.0 * math.pi / n) + math.pi / 8.0
    return torch.stack((radius_x * torch.cos(theta), radius_y * torch.sin(theta)), dim=-1)


def _sample_polygon_uniform(points01: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """Uniform samples from a regular/affine-regular polygon by equal-area fan triangles."""
    n_tri = vertices.shape[0]
    tri_idx = torch.clamp((points01[:, 0] * n_tri).long(), max=n_tri - 1)
    local_u = points01[:, 0] * n_tri - tri_idx.to(points01.dtype)
    local_v = points01[:, 1]
    a = vertices[tri_idx]
    b = vertices[(tri_idx + 1) % n_tri]
    root_u = torch.sqrt(torch.clamp(local_u, min=0.0, max=1.0))
    return root_u[:, None] * ((1.0 - local_v)[:, None] * a + local_v[:, None] * b)


def build_emitter_tangent_frame(
    forward: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs a canonical orthonormal coordinate frame (right, up, forward)."""
    f_norm = F.normalize(forward, dim=-1, eps=_EPS)
    world_up = torch.zeros_like(f_norm)
    world_up[..., 1] = 1.0
    alternate_up = torch.zeros_like(f_norm)
    alternate_up[..., 2] = 1.0
    use_alt = torch.abs((f_norm * world_up).sum(dim=-1, keepdim=True)) > 0.98
    up_ref = torch.where(use_alt, alternate_up, world_up)
    right = F.normalize(torch.cross(f_norm, up_ref, dim=-1), dim=-1, eps=_EPS)
    up = F.normalize(torch.cross(right, f_norm, dim=-1), dim=-1, eps=_EPS)
    return right, up, f_norm


def generate_emitter_samples_3d(
    light_pos: torch.Tensor,
    light_aim: torch.Tensor,
    emitter: EmitterSpec,
    num_samples: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates camera-space 3D sample points distributed across the physical aperture face."""
    if light_pos.ndim != 2 or light_pos.shape[-1] != 3:
        raise ValueError("light_pos must be [B,3]")
    if light_aim.ndim != 2 or light_aim.shape[-1] != 3:
        raise ValueError("light_aim must be [B,3]")
    if light_pos.shape[0] != light_aim.shape[0]:
        raise ValueError("light_pos/light_aim batch mismatch")
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")

    b = light_pos.shape[0]
    device, dtype = light_pos.device, light_pos.dtype
    samples01 = _hammersley_2d(num_samples, device, dtype)

    right, up, _ = build_emitter_tangent_frame(light_aim - light_pos)

    if emitter.shape == "rectangle":
        local_x = (samples01[:, 0] - 0.5) * emitter.width_m
        local_y = (samples01[:, 1] - 0.5) * emitter.height_m
        local = torch.stack((local_x, local_y), dim=-1)
    elif emitter.shape == "circle":
        u_clamped = torch.clamp(samples01[:, 0], min=0.0, max=1.0)
        r_in2 = emitter.inner_radius_m * emitter.inner_radius_m
        r_out2 = emitter.radius_m * emitter.radius_m
        r = torch.sqrt(r_in2 + (r_out2 - r_in2) * u_clamped)
        theta = 2.0 * math.pi * samples01[:, 1]
        local = torch.stack((r * torch.cos(theta), r * torch.sin(theta)), dim=-1)
    elif emitter.shape == "octagon":
        verts = _regular_polygon_vertices(
            8,
            emitter.width_m * 0.5,
            emitter.height_m * 0.5,
            device,
            dtype,
        )
        local = _sample_polygon_uniform(samples01, verts)
    else:
        raise ValueError(f"Unsupported emitter shape: {emitter.shape}")

    offsets = local[:, None, 0] * right[:, None, :] + local[:, None, 1] * up[:, None, :]
    points = light_pos[:, None, :] + offsets
    weights = torch.full((b, num_samples), 1.0 / num_samples, device=device, dtype=dtype)
    return points, weights


def _evaluate_occlusion(
    sampled_depth: torch.Tensor,
    ray_depth: torch.Tensor,
    receiver_depth: torch.Tensor,
    light_depth: torch.Tensor,
    margin: torch.Tensor,
    max_thickness_m: Optional[float] = None,
) -> torch.Tensor:
    """
    True iff sampled scene depth geometrically occludes the ray within physical object thickness.
    Setting `max_thickness_m = 0.45` prevents 2.5D depth sheets (e.g. a shoulder at Z=-2.0m or a
    candlestick) from acting as infinitely thick solid blocks all the way to Z=-5.0m.
    """
    min_bound = torch.minimum(receiver_depth, light_depth) - margin
    max_bound = ray_depth - margin
    is_in_front_of_ray = sampled_depth <= max_bound
    is_behind_frontmost = sampled_depth >= min_bound
    valid_hit = (
        torch.isfinite(sampled_depth)
        & is_in_front_of_ray
        & is_behind_frontmost
    )
    if max_thickness_m is not None:
        valid_hit = valid_hit & ((ray_depth - sampled_depth) <= float(max_thickness_m))
    return valid_hit


def metric_shadow_visibility_points(
    surface_xyz: torch.Tensor,
    scene_depth: torch.Tensor,
    light_points: torch.Tensor,
    camera: CameraModel,
    *,
    steps: int = 28,
    max_distance_m: Optional[float] = None,
    depth_bias_m: float = 0.018,
    ray_start_m: float = 0.032,
    ray_end_m: float = 0.008,
    max_thickness_m: Optional[float] = 0.45,
    dither: bool = True,
) -> torch.Tensor:
    """Vectorized point-source shadow raymarching for multiple emitter samples."""
    surface_xyz = _ensure_bhw3(surface_xyz, "surface_xyz")
    scene_depth = _ensure_bhw(scene_depth, "scene_depth")
    if light_points.ndim != 3 or light_points.shape[-1] != 3:
        raise ValueError("light_points must be [B,S,3]")
    if surface_xyz.shape[:3] != scene_depth.shape:
        raise ValueError("surface_xyz and scene_depth spatial shapes must match")
    if surface_xyz.shape[0] != light_points.shape[0]:
        raise ValueError("surface/light batch mismatch")
    if steps < 2:
        raise ValueError("steps must be >= 2")

    b, h, w, _ = surface_xyz.shape
    s = light_points.shape[1]
    p = surface_xyz

    to_light = light_points[:, :, None, None, :] - p[:, None, :, :, :]
    distance = torch.linalg.vector_norm(to_light, dim=-1)
    direction = to_light / torch.clamp(distance[..., None], min=_EPS)

    if max_distance_m is None:
        ray_length = distance
    else:
        ray_length = torch.minimum(
            distance,
            torch.full_like(distance, float(max_distance_m)),
        )

    receiver_depth = (-p[..., 2])[:, None]
    light_depth = (-light_points[..., 2])[:, :, None, None]
    finite_receiver = (
        torch.isfinite(p).all(-1)[:, None]
        & torch.isfinite(scene_depth)[:, None]
        & (receiver_depth > camera.near_m)
        & (receiver_depth < camera.far_m)
        & (ray_length > ray_start_m + ray_end_m)
    )

    occluded = torch.zeros((b, s, h, w), device=p.device, dtype=torch.bool)

    start = torch.as_tensor(ray_start_m, device=p.device, dtype=p.dtype)
    end = torch.as_tensor(ray_end_m, device=p.device, dtype=p.dtype)
    alpha0 = torch.clamp(start / torch.clamp(ray_length, min=_EPS), 0.0, 0.2)
    alpha1 = 1.0 - torch.clamp(end / torch.clamp(ray_length, min=_EPS), 0.0, 0.2)

    if dither:
        py = torch.arange(h, device=p.device, dtype=p.dtype)
        px = torch.arange(w, device=p.device, dtype=p.dtype)
        grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
        ign = ((52.9829189 * ((grid_x * 0.06711056 + grid_y * 0.00583715) % 1.0)) % 1.0)
        # Per-sample golden-ratio phase shift so each emitter sample dithers independently
        s_phase = (torch.arange(s, device=p.device, dtype=p.dtype) * 0.6180339887).view(1, s, 1, 1)
        step_dither = ((ign.unsqueeze(0).unsqueeze(0) + s_phase) % 1.0) - 0.5
    else:
        step_dither = 0.0

    step_delta = 1.0 / float(steps)
    base_margin = torch.as_tensor(depth_bias_m, device=p.device, dtype=p.dtype)

    receiver_bs = receiver_depth.expand(b, s, h, w).reshape(b * s, h, w)
    light_bs = light_depth.expand(b, s, h, w).reshape(b * s, h, w)

    for step_idx in range(steps):
        if isinstance(step_dither, torch.Tensor):
            a = torch.clamp((step_idx + 0.5 + step_dither) * step_delta, 0.0, 1.0)
        else:
            a = min(max((step_idx + 0.5) * step_delta, 0.0), 1.0)
        t = alpha0 + (alpha1 - alpha0) * a
        q = p[:, None, :, :, :] + direction * (ray_length * t)[..., None]

        q_bs = q.reshape(b * s, h, w, 3)
        grid, ray_depth, in_bounds = project_camera_points(q_bs, camera)

        depth_bs = scene_depth[:, None, :, :].expand(b, s, h, w).reshape(b * s, h, w)
        sampled_depth = _sample_depth(depth_bs, grid, in_bounds)

        margin = base_margin + 0.002 * torch.clamp(ray_depth, min=0.0)

        hit = _evaluate_occlusion(
            sampled_depth=sampled_depth,
            ray_depth=ray_depth,
            receiver_depth=receiver_bs,
            light_depth=light_bs,
            margin=margin,
            max_thickness_m=max_thickness_m,
        )
        hit = hit & in_bounds
        hit = hit.reshape(b, s, h, w) & finite_receiver
        occluded |= hit

        if (occluded | ~finite_receiver).all():
            break

    return torch.where(
        finite_receiver,
        (~occluded).to(p.dtype),
        torch.ones((b, s, h, w), device=p.device, dtype=p.dtype),
    )


def metric_shadow_visibility_point(
    surface_xyz: torch.Tensor,
    scene_depth: torch.Tensor,
    light_pos: torch.Tensor,
    camera: CameraModel,
    *,
    steps: int = 28,
    max_distance_m: Optional[float] = None,
    depth_bias_m: float = 0.018,
    ray_start_m: float = 0.032,
    ray_end_m: float = 0.008,
    max_thickness_m: Optional[float] = 0.45,
    dither: bool = False,
) -> torch.Tensor:
    """Scalar point-light visibility wrapper."""
    if light_pos.ndim != 2 or light_pos.shape[-1] != 3:
        raise ValueError("light_pos must be [B,3]")
    return metric_shadow_visibility_points(
        surface_xyz,
        scene_depth,
        light_pos[:, None, :],
        camera,
        steps=steps,
        max_distance_m=max_distance_m,
        depth_bias_m=depth_bias_m,
        ray_start_m=ray_start_m,
        ray_end_m=ray_end_m,
        max_thickness_m=max_thickness_m,
        dither=dither,
    )[:, 0]


def _bilateral_pass_1d_dilated(
    vis_bhw: torch.Tensor,
    scene_depth_bhw: torch.Tensor,
    sigma_spatial: float,
    inv_2sd2: float,
    dilation: int = 1,
) -> torch.Tensor:
    """2-pass separable 5-tap dilated depth-guided bilateral filter."""
    offsets = (-2, -1, 0, 1, 2)
    sp_weights = tuple(
        math.exp(-(o * o) / (2.0 * max(float(sigma_spatial), 1e-4) ** 2))
        for o in offsets
    )
    pad = 2 * int(dilation)
    _, h_dim, w_dim = vis_bhw.shape

    # Horizontal dilated pass
    v_pad_h = F.pad(vis_bhw, (pad, pad, 0, 0), mode="replicate")
    d_pad_h = F.pad(scene_depth_bhw, (pad, pad, 0, 0), mode="replicate")
    num_h = torch.zeros_like(vis_bhw)
    den_h = torch.zeros_like(vis_bhw)
    for o, ws in zip(offsets, sp_weights):
        shift = pad + o * int(dilation)
        v_s = v_pad_h[:, :, shift : shift + w_dim]
        d_s = d_pad_h[:, :, shift : shift + w_dim]
        w = ws * torch.exp(-((d_s - scene_depth_bhw) ** 2) * inv_2sd2)
        num_h = num_h + v_s * w
        den_h = den_h + w
    vis_h = num_h / torch.clamp(den_h, min=_EPS)

    # Vertical dilated pass
    v_pad_v = F.pad(vis_h, (0, 0, pad, pad), mode="replicate")
    d_pad_v = F.pad(scene_depth_bhw, (0, 0, pad, pad), mode="replicate")
    num_v = torch.zeros_like(vis_bhw)
    den_v = torch.zeros_like(vis_bhw)
    for o, ws in zip(offsets, sp_weights):
        shift = pad + o * int(dilation)
        v_s = v_pad_v[:, shift : shift + h_dim, :]
        d_s = d_pad_v[:, shift : shift + h_dim, :]
        w = ws * torch.exp(-((d_s - scene_depth_bhw) ** 2) * inv_2sd2)
        num_v = num_v + v_s * w
        den_v = den_v + w
    return torch.clamp(num_v / torch.clamp(den_v, min=_EPS), 0.0, 1.0)


def depth_guided_shadow_denoise(
    vis_bhw: torch.Tensor,
    scene_depth_bhw: torch.Tensor,
    sigma_spatial: float = 1.8,
    sigma_depth_m: float = 0.045,
) -> torch.Tensor:
    """
    Multi-scale À-Trous (stride 1 + stride 3) edge-preserving depth-guided bilateral filter.
    Eliminates both high-frequency IGN dither AND medium-frequency raymarching step bands
    across background walls while strictly preserving sharp occlusion boundaries.
    """
    inv_2sd2 = 1.0 / (2.0 * max(float(sigma_depth_m), 1e-4) ** 2)
    vis = _bilateral_pass_1d_dilated(vis_bhw, scene_depth_bhw, sigma_spatial, inv_2sd2, dilation=1)
    vis = _bilateral_pass_1d_dilated(vis, scene_depth_bhw, sigma_spatial, inv_2sd2, dilation=3)
    return vis


def metric_shadow_visibility_area(
    surface_xyz: torch.Tensor,
    scene_depth: torch.Tensor,
    light_pos: torch.Tensor,
    light_aim: torch.Tensor,
    emitter: EmitterSpec,
    camera: CameraModel,
    *,
    num_emitter_samples: int = 8,
    ray_steps: int = 28,
    sample_chunk: int = 4,
    max_distance_m: Optional[float] = None,
    depth_bias_m: float = 0.018,
    ray_start_m: float = 0.032,
    ray_end_m: float = 0.008,
    max_thickness_m: Optional[float] = 0.45,
    dither: bool = True,
) -> torch.Tensor:
    """Area-emitter soft shadow visibility via multi-sample aperture integration."""
    surface_xyz = _ensure_bhw3(surface_xyz, "surface_xyz")
    scene_depth = _ensure_bhw(scene_depth, "scene_depth")
    emitter_points, emitter_weights = generate_emitter_samples_3d(
        light_pos, light_aim, emitter, num_emitter_samples
    )

    b, h, w, _ = surface_xyz.shape
    visibility = torch.zeros(
        (b, h, w), device=surface_xyz.device, dtype=surface_xyz.dtype
    )

    for s0 in range(0, num_emitter_samples, sample_chunk):
        s1 = min(num_emitter_samples, s0 + sample_chunk)
        chunk_points = emitter_points[:, s0:s1, :]
        chunk_weights = emitter_weights[:, s0:s1]

        chunk_vis = metric_shadow_visibility_points(
            surface_xyz,
            scene_depth,
            chunk_points,
            camera,
            steps=ray_steps,
            max_distance_m=max_distance_m,
            depth_bias_m=depth_bias_m,
            ray_start_m=ray_start_m,
            ray_end_m=ray_end_m,
            max_thickness_m=max_thickness_m,
            dither=dither,
        )
        chunk_vis = chunk_vis * chunk_weights[:, :, None, None]
        visibility = visibility + chunk_vis.sum(dim=1)

    visibility = torch.clamp(visibility, 0.0, 1.0)
    if dither:
        visibility = depth_guided_shadow_denoise(visibility, scene_depth)
    return visibility


def screen_space_directional_shadow_metric(
    surface_xyz: torch.Tensor,
    scene_depth: torch.Tensor,
    light_pos: torch.Tensor,
    light_aim: torch.Tensor,
    modifier: str = "Octabox (Soft)",
    *,
    vertical_fov_deg: float = 40.0,
    aspect: Optional[float] = None,
    num_emitter_samples: int = 8,
    ray_steps: int = 28,
    sample_chunk: int = 4,
    max_distance_m: Optional[float] = None,
    depth_bias_m: float = 0.018,
    ray_start_m: float = 0.032,
    ray_end_m: float = 0.008,
    max_thickness_m: Optional[float] = 0.45,
    dither: bool = True,
) -> torch.Tensor:
    """High-level entry point for StudioLight shader engine."""
    emitter = MODIFIER_EMITTERS.get(modifier, MODIFIER_EMITTERS["Octabox (Soft)"])
    if aspect is None:
        aspect = surface_xyz.shape[2] / max(surface_xyz.shape[1], 1)

    camera = CameraModel(
        vertical_fov_deg=vertical_fov_deg,
        aspect=float(aspect),
        image_y_down=True,
    )
    return metric_shadow_visibility_area(
        surface_xyz,
        scene_depth,
        light_pos,
        light_aim,
        emitter,
        camera,
        num_emitter_samples=num_emitter_samples,
        ray_steps=ray_steps,
        sample_chunk=sample_chunk,
        max_distance_m=max_distance_m,
        depth_bias_m=depth_bias_m,
        ray_start_m=ray_start_m,
        ray_end_m=ray_end_m,
        max_thickness_m=max_thickness_m,
        dither=dither,
    )
