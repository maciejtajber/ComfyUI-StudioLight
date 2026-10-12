"""
ComfyUI-StudioLight — Photometric Shader & Relighting Engine
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Calculates physical deferred lighting across surface normals (diffuse + specular)
and generates visual VFX probes with an integrated 2D studio radar compass.
"""

import os

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, Tuple
from .color_utils import kelvin_to_rgb, MODIFIER_PRESETS
from .metric_shadow import (
    _hammersley_2d,
    _regular_polygon_vertices,
    _sample_polygon_uniform,
    screen_space_directional_shadow_metric,
)


def _safe_quantile(t: torch.Tensor, q: float, max_samples: int = 524288) -> torch.Tensor:
    """
    Computes tensor quantile safely across arbitrary resolutions (including 4K/8K batches).
    PyTorch's torch.quantile raises a RuntimeError above 2^24 (~16.7M) elements; deterministic
    strided subsampling above max_samples prevents OOM/limit crashes while preserving accuracy.
    """
    flat = t.reshape(-1)
    n = flat.numel()
    if n > max_samples:
        step = max(1, n // max_samples)
        flat = flat[::step]
    return torch.quantile(flat, q)


def get_preferred_device() -> torch.device:
    """Returns fastest available hardware accelerator (CUDA -> MPS -> CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def azimuth_elevation_to_vector(azimuth_deg: float, elevation_deg: float) -> torch.Tensor:
    """
    Converts Azimuth (-180° to +180°) and Elevation (-30° to +90°) to a 3D unit vector.
    Conventions (Camera Space):
      - Camera looks down -Z towards subject at (0, 0, 0).
      - X+: Camera Right
      - Y+: Camera Up (OpenGL convention)
      - Z+: Pointing towards camera lens
      - Azimuth 0° = Front / On-axis
      - Azimuth +45° = Camera Right Rembrandt Key
      - Azimuth -45° = Camera Left
      - Azimuth +/-180° = Direct Backlight
      - Elevation 0° = Eye level
      - Elevation -30° = Low footlight / floor uplight
      - Elevation +90° = Overhead top light
    """
    theta = math.radians(float(azimuth_deg))
    phi = math.radians(float(elevation_deg))

    # Spherical coordinates
    lx = math.sin(theta) * math.cos(phi)
    ly = math.sin(phi)
    lz = math.cos(theta) * math.cos(phi)

    vec = torch.tensor([lx, ly, lz], dtype=torch.float32)
    norm = torch.norm(vec)
    if norm > 1e-6:
        return vec / norm
    return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)


def compute_cavity_ao_map(normals: torch.Tensor, strength: float = 1.5) -> torch.Tensor:
    """
    Derives Ambient Occlusion / Cavity map directly from normal field divergence.
    Detects concave folds, creases under chin/jaw, and seam boundaries.
    normals: [B, H, W, 3] in [-1, 1] range.
    returns: [B, H, W, 1] cavity multiplier (1.0 = open surface, 0.0 = deep crevice/seam)
    """
    B, H, W, _ = normals.shape
    device = normals.device
    dtype = normals.dtype

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3) / 8.0

    nx = normals[..., 0:1].permute(0, 3, 1, 2).contiguous()
    ny = normals[..., 1:2].permute(0, 3, 1, 2).contiguous()

    dnx_dx = F.conv2d(nx, kx, padding=1)
    dny_dy = F.conv2d(ny, ky, padding=1)

    # Negative divergence indicates a concave valley / seam
    curvature = (dnx_dx + dny_dy).permute(0, 2, 3, 1).contiguous()

    # Crevices: curvature < 0 (normals point inward towards each other)
    cavity = torch.clamp(1.0 + curvature * float(strength), min=0.0, max=1.0)
    return cavity


def render_sphere_normals(resolution: int = 256, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates 3D hemisphere normal map and antialiased circular mask."""
    if device is None:
        device = get_preferred_device()

    res = int(resolution)
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, res, device=device),
        torch.linspace(-1.0, 1.0, res, device=device),
        indexing="ij"
    )
    r2 = x * x + y * y
    mask = (r2 <= 1.0).float()

    ny = -y
    nx = x
    nz = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))

    normals = torch.stack([nx, ny, nz], dim=-1) * mask.unsqueeze(-1)
    return normals.unsqueeze(0), mask.unsqueeze(0).unsqueeze(-1)



def generate_emitter_tangent_frame(light_dir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs an orthonormal coordinate frame [u_tangent, v_tangent, w_aim]
    for a light fixture oriented along light_dir pointing towards scene center.
    w_aim: Points forward from fixture face towards subject (-light_dir).
    u_tangent: Horizontal tangent (fixture right).
    v_tangent: Vertical tangent (fixture up).
    """
    device = light_dir.device
    dtype = light_dir.dtype

    l_unit = light_dir / (torch.norm(light_dir) + 1e-7)
    w_aim = -l_unit

    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    if torch.abs(torch.dot(w_aim, up)) > 0.92:
        up = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

    u_tangent = torch.linalg.cross(w_aim, up)
    u_tangent = u_tangent / (torch.norm(u_tangent) + 1e-7)
    v_tangent = torch.linalg.cross(w_aim, u_tangent)
    v_tangent = v_tangent / (torch.norm(v_tangent) + 1e-7)

    return u_tangent, v_tangent, w_aim


def generate_emitter_samples_2d(preset: Dict[str, Any], num_samples: int = 16) -> torch.Tensor:
    """
    Generates deterministic low-discrepancy 2D metric offsets (u, v) in meters across the emitter face.
    Returns: [num_samples, 2] tensor.
    """
    geom = preset.get("geometry", "circle")
    offsets = []
    golden_angle = 2.399963229728653

    if geom == "rectangle":
        w = float(preset.get("width_m", 0.30))
        h = float(preset.get("height_m", 1.20))
        grid_n = int(math.isqrt(num_samples))
        if grid_n * grid_n < num_samples:
            grid_n += 1
        for i in range(grid_n):
            for j in range(grid_n):
                if len(offsets) >= num_samples:
                    break
                u = w * (-0.5 + (i + 0.5) / float(grid_n))
                v = h * (-0.5 + (j + 0.5) / float(grid_n))
                offsets.append([u, v])
    elif geom == "beauty_dish":
        r_out = float(preset.get("radius_m", 0.275))
        r_in = r_out * float(preset.get("deflector_ratio", 0.36))
        for k in range(num_samples):
            frac = (k + 0.5) / float(num_samples)
            r = math.sqrt(r_in * r_in + (r_out * r_out - r_in * r_in) * frac)
            th = k * golden_angle
            offsets.append([r * math.cos(th), r * math.sin(th)])
    elif geom == "octagon":
        r_out = float(preset.get("radius_m", 0.70))
        pts01 = _hammersley_2d(num_samples, torch.device("cpu"), torch.float32)
        verts = _regular_polygon_vertices(8, r_out, r_out, torch.device("cpu"), torch.float32)
        return _sample_polygon_uniform(pts01, verts)
    else:
        r_out = float(preset.get("radius_m", 0.70))
        for k in range(num_samples):
            frac = (k + 0.5) / float(num_samples)
            r = r_out * math.sqrt(frac)
            th = k * golden_angle
            offsets.append([r * math.cos(th), r * math.sin(th)])

    return torch.tensor(offsets, dtype=torch.float32)


def evaluate_fixture_aperture_footprint(
    refl_rays: torch.Tensor,
    surf_pos: torch.Tensor,
    p_light: torch.Tensor,
    u_tangent: torch.Tensor,
    v_tangent: torch.Tensor,
    w_aim: torch.Tensor,
    preset: Dict[str, Any],
    roughness: float = 0.08,
) -> torch.Tensor:
    """
    Intersects reflection rays with the 3D fixture aperture plane and evaluates
    the geometric aperture footprint (Stripbox rectangle, Beauty Dish parabolic ring,
    Octabox faceted disc, Fresnel circle).
    Returns specular reflection intensity in [0, 1] range.
    """
    denom = torch.sum(refl_rays * w_aim.view(1, 1, 1, 3), dim=-1, keepdim=True)
    numer = torch.sum((p_light.view(1, 1, 1, 3) - surf_pos) * w_aim.view(1, 1, 1, 3), dim=-1, keepdim=True)
    t = numer / (denom - 1e-6)
    valid_hit = (t > 0.0) & (denom < -0.01)

    p_hit = surf_pos + t * refl_rays
    diff = p_hit - p_light.view(1, 1, 1, 3)
    u = torch.sum(diff * u_tangent.view(1, 1, 1, 3), dim=-1, keepdim=True)
    v = torch.sum(diff * v_tangent.view(1, 1, 1, 3), dim=-1, keepdim=True)

    geom = preset.get("geometry", "circle")
    blur = float(roughness) * 0.35

    if geom == "rectangle":
        w = float(preset.get("width_m", 0.30))
        h = float(preset.get("height_m", 1.20))
        ew = w * 0.5 + blur
        eh = h * 0.5 + blur
        du = torch.abs(u) / ew
        dv = torch.abs(v) / eh
        d_box = torch.maximum(du, dv)
        edge_pow = max(2.0, 12.0 / (1.0 + float(roughness) * 12.0))
        spec = torch.clamp(1.0 - torch.pow(d_box, edge_pow), 0.0, 1.0)
    elif geom == "beauty_dish":
        r = torch.sqrt(u * u + v * v)
        rm = float(preset.get("radius_m", 0.275)) + blur
        r_norm = r / rm
        edge_pow = max(2.0, 8.0 / (1.0 + float(roughness) * 10.0))
        outer = torch.clamp(1.0 - torch.pow(r_norm, edge_pow), 0.0, 1.0)
        defl_r = float(preset.get("radius_m", 0.275)) * float(preset.get("deflector_ratio", 0.36))
        trans = float(preset.get("deflector_transmission", 0.12))
        inner_shadow = torch.clamp((r / defl_r - 0.6) * 3.0, 0.0, 1.0)
        inner = trans + (1.0 - trans) * inner_shadow
        spec = outer * inner
    elif geom == "octagon":
        r = torch.sqrt(u * u + v * v)
        th = torch.atan2(v, u)
        th_mod = torch.remainder(th, math.pi / 4.0) - math.pi / 8.0
        r_oct = float(preset.get("radius_m", 0.70)) * math.cos(math.pi / 8.0) / torch.cos(th_mod) + blur
        edge_pow = max(2.0, 6.0 / (1.0 + float(roughness) * 8.0))
        spec = torch.clamp(1.0 - torch.pow(r / r_oct, edge_pow), 0.0, 1.0)
    else:
        r = torch.sqrt(u * u + v * v)
        rm = float(preset.get("radius_m", 0.15)) + blur
        edge_pow = max(2.0, 8.0 / (1.0 + float(roughness) * 10.0))
        spec = torch.clamp(1.0 - torch.pow(r / rm, edge_pow), 0.0, 1.0)

    gain = 1.0 / (1.0 + float(roughness) * 3.0)
    return spec * gain * valid_hit.float()


AIM_TARGET_OPTIONS = [
    "Upper",
    "Center",
    "Lower",
]


def evaluate_beam_profile(cos_e: torch.Tensor, preset: Dict[str, Any]) -> torch.Tensor:
    """
    Evaluates a smooth, calibrated studio photometric beam profile from emitter off-axis cosine `cos_e`.

    Parameter-Semantics & Calibration Note:
    ---------------------------------------
    - `beam_angle_deg` (theta_beam) defines the Full-Width at Half-Maximum (FWHM / 50% intensity cone):
      at off-axis half-angle `theta_half = theta_beam / 2`, the profile evaluates to exactly `0.50`
      (replacing the legacy hard 0.0 step cutoff at `theta_half`).
    - Because the penumbra skirt extends smoothly beyond `theta_half` out to `theta_field`, the visible
      illuminated footprint is wider than the legacy hard step cutoff for identical `beam_angle_deg`.
    - Continuity: Inside `[0, theta_half]`, `skirt == 1.0` and `skirt'(theta_half) == 0.0`; across
      `[theta_half, theta_field]`, the cubic Hermite polynomial `3*t^2 - 2*t^3` joins with continuous
      value (C^0) and continuous first derivative (C^1) at both `theta_half` and `theta_field`, reaching
      exact `0.0` for `theta >= theta_field`.
    - Note: This is a smooth, calibrated photometric beam profile for virtual studio relighting and
      modifier differentiation, not a physical wave-optics Fresnel-lens simulation.
    """
    beam_angle = float(preset.get("beam_angle_deg", 90.0))
    edge_softness = float(preset.get("edge_softness", 0.15))

    theta_half_deg = min(max(beam_angle * 0.5, 4.0), 82.0)
    cos_half = math.cos(math.radians(theta_half_deg))
    # Exponent chosen so cos_e == cos_half yields exactly 0.50 (FWHM convention)
    p_beam = math.log(0.5) / math.log(max(cos_half, 1e-4))

    # Outer field-angle penumbra skirt: 1.0 inside FWHM cone, C^1 Hermite rolloff from theta_half to theta_field
    theta_field_deg = min(88.0, theta_half_deg * (1.85 + 1.20 * edge_softness) + 12.0)
    cos_field = math.cos(math.radians(theta_field_deg))
    skirt_t = torch.clamp((cos_e - cos_field) / max(cos_half - cos_field, 1e-4), 0.0, 1.0)
    skirt = skirt_t * skirt_t * (3.0 - 2.0 * skirt_t)

    lobe = torch.pow(torch.clamp(cos_e, 1e-6, 1.0), p_beam)
    return torch.clamp(lobe * skirt, 0.0, 1.0)


def reconstruct_metric_surface(
    depth: torch.Tensor,
    H: int,
    W: int,
    vertical_fov_deg: float = 40.0,
    depth_relief_scale: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Reconstructs calibrated camera-space 3D surface positions `p_surf = (X, Y, Z)` from a relative depth map.

    Assumptions & Properties:
    -------------------------
    1. Calibrated Virtual Studio Coordinate Frame: Because monocular depth maps (`DepthAnythingV2`, `Marigold`)
       provide uncalibrated relative depth `d_norm in [0, 1]` rather than physical range data, coordinates
       expressed in meters represent a calibrated virtual studio reconstruction anchored at `Z_subj = -2.0m`
       under `vertical_fov_deg`, not guaranteed real-world physical distances.
    2. Order-Preserving Subject-Relief Compression (`dZ / dd_norm > 0` everywhere): Relative front-to-back
       depth ordering is strictly preserved across `[0, 1]`, while foreground subject relief is intentionally
       compressed by `depth_relief_scale` around `Z = -2.0m` (mitigating exaggerated monocular torso-to-face
       depth stretch and false upward collar shadows) and smoothly blended via a closed-form C^inf softplus
       antiderivative into the full `3.0m` background depth slope (`depth_relief_scale=1.0` reproduces the
       uncompressed linear ramp).
    3. Adaptive Subject Anchoring: Estimates the foreground/background split `d_split` and subject median
       `d_subj` from the central framing region rather than relying solely on fixed `d_norm` thresholds.
    """
    device = depth.device
    dtype = depth.dtype
    d = depth
    if d.shape[-1] != 1 and len(d.shape) == 4:
        d = d[..., 0:1]
    elif len(d.shape) == 3:
        d = d.unsqueeze(-1)
    d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    B = d.shape[0]
    if d.shape[1] != H or d.shape[2] != W:
        d_perm = d.permute(0, 3, 1, 2).contiguous()
        d = F.interpolate(d_perm, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1).contiguous()

    aspect = float(W) / float(H)
    tan_half_fov = math.tan(math.radians(float(vertical_fov_deg)) * 0.5)
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(1.0, -1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
    y_grid = y_coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

    d_flat = d.reshape(-1)
    if d_flat.numel() > 1000:
        d_min = _safe_quantile(d_flat, 0.01)
        d_max = _safe_quantile(d_flat, 0.99)
    else:
        d_min = torch.min(d)
        d_max = torch.max(d)
    d_range = d_max - d_min

    if float(d_range) < 1e-4:
        d_norm = torch.ones_like(d)
        z_pos = torch.full_like(d, -2.0)
        d_split_val = 0.5
    else:
        d_norm = torch.clamp((d - d_min) / (d_range + 1e-6), 0.0, 1.0)
        cy0, cy1 = max(0, int(0.10 * H)), min(H, max(1, int(0.88 * H)))
        cx0, cx1 = max(0, int(0.20 * W)), min(W, max(1, int(0.80 * W)))
        central = d_norm[:, cy0:cy1, cx0:cx1, 0].reshape(-1)
        valid_c = central[central > 0.05]
        if valid_c.numel() > 16:
            d_split_val = float(torch.clamp(_safe_quantile(valid_c, 0.20), 0.18, 0.50))
            fg_vals = valid_c[valid_c >= d_split_val]
            d_subj_val = float(torch.median(fg_vals)) if fg_vals.numel() > 0 else 0.65
        else:
            d_split_val = 0.35
            d_subj_val = 0.65

        scale = min(max(float(depth_relief_scale), 0.10), 1.0)
        if scale >= 0.995:
            z_pos = -(1.5 + (1.0 - d_norm) * 3.0)
        else:
            s_fg = 3.0 * scale
            s_bg = 3.0
            tau = 0.06
            u_d = (d_norm - d_split_val) / tau
            u_subj = torch.tensor((d_subj_val - d_split_val) / tau, device=device, dtype=dtype)
            delta_z = s_bg * (d_norm - d_subj_val) + (s_fg - s_bg) * tau * (
                F.softplus(u_d) - F.softplus(u_subj)
            )
            z_pos = torch.clamp(-2.0 + delta_z, min=-6.0, max=-0.35)

    x_pos = x_grid * tan_half_fov * aspect * (-z_pos)
    y_pos = y_grid * tan_half_fov * (-z_pos)
    p_surf = torch.cat([x_pos, y_pos, z_pos], dim=-1)
    return p_surf, d_norm, d_split_val


def resolve_aim_target(
    p_surf: Optional[torch.Tensor],
    d_norm: Optional[torch.Tensor],
    normals: Optional[torch.Tensor] = None,
    aim_mode: str = "Upper",
    d_split: float = 0.35,
    target_pos: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Resolves the 3D calibrated camera-space aim point `[1, 3]` `(X, Y, Z)` for studio light fixtures.

    Scale-Invariant Auto-Aim & Confidence Fallback:
    -----------------------------------------------
    - Evaluates surface normal curvature `|grad_c N|` and depth step edges `|grad_c d|` on a fixed
      canonical `128x128` grid so gradient magnitudes and exponential edge suppression `exp(-8 * |grad_c d|)`
      are invariant to image resolution (`256x256` through `4096x4096`) and resilient to pixel-level noise.
    - Confidence Gate (`p90_score >= 0.015`): Requires genuine 3D surface curvature in the upper portrait
      region (satisfied by both detailed human faces and smooth curved synthetic/android heads, while
      rejecting flat architectural walls where `|grad_c N| < 0.005`).
    - When curvature confidence is low (e.g. a flat upper background wall above a lower tabletop object),
      automatically falls back to the full-frame foreground Subject Centroid.
    """
    if p_surf is not None:
        device = p_surf.device
        dtype = p_surf.dtype
    elif normals is not None:
        device = normals.device
        dtype = normals.dtype
    elif device is None:
        device = torch.device("cpu")

    if target_pos is not None:
        return target_pos.to(device=device, dtype=dtype).reshape(-1, 3)[:1]

    default_center = torch.tensor([[0.0, 0.0, -2.0]], device=device, dtype=dtype)
    if p_surf is None or d_norm is None or aim_mode == "Legacy Center Target":
        return default_center

    B, H, W, _ = p_surf.shape
    if H < 16 or W < 16:
        return default_center

    dn = d_norm[0, :, :, 0]
    ps = p_surf[0]
    fg_mask = (dn >= d_split) & (dn > 0.10)

    def _region_fg_median(y0_f: float, y1_f: float, x0_f: float, x1_f: float) -> Optional[torch.Tensor]:
        y0, y1 = max(0, int(y0_f * H)), min(H, max(1, int(y1_f * H)))
        x0, x1 = max(0, int(x0_f * W)), min(W, max(1, int(x1_f * W)))
        sub_fg = fg_mask[y0:y1, x0:x1]
        min_px = max(8, int(0.015 * (y1 - y0) * (x1 - x0)))
        if int(sub_fg.sum().item()) < min_px:
            return None
        dn_sub = dn[y0:y1, x0:x1]
        fg_d = dn_sub[sub_fg]
        q90 = float(_safe_quantile(fg_d, 0.90).item())
        q20 = float(_safe_quantile(fg_d, 0.20).item())
        if q90 - q20 > 0.15:
            closer_mask = sub_fg & (dn_sub >= 0.5 * (q90 + q20))
            if int(closer_mask.sum().item()) >= 4:
                sub_fg = closer_mask
        pts = ps[y0:y1, x0:x1][sub_fg]
        return torch.median(pts, dim=0).values.view(1, 3)

    if aim_mode in ("Upper", "Auto (Face / Upper Focus)"):
        y0, y1 = max(0, int(0.05 * H)), min(H, max(1, int(0.30 * H)))
        x0, x1 = max(0, int(0.22 * W)), min(W, max(1, int(0.78 * W)))
        sub_fg = fg_mask[y0:y1, x0:x1]
        min_px = max(8, int(0.015 * (y1 - y0) * (x1 - x0)))
        is_uniform_plane = float(fg_mask.float().mean().item()) > 0.92
        if int(sub_fg.sum().item()) >= min_px:
            if normals is not None and normals.shape[1] == H and normals.shape[2] == W:
                # Evaluate finite differences on a fixed canonical 128x128 grid for resolution invariance
                c_res = 128
                n_c = F.interpolate(
                    normals[0:1].permute(0, 3, 1, 2).contiguous(),
                    size=(c_res, c_res),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()
                n_c = n_c / (torch.norm(n_c, dim=-1, keepdim=True) + 1e-7)
                d_c = F.interpolate(
                    d_norm[0:1].permute(0, 3, 1, 2).contiguous(),
                    size=(c_res, c_res),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()

                dn_dx = torch.norm(n_c[:, :, 1:, :] - n_c[:, :, :-1, :], dim=-1, keepdim=True)
                dn_dy = torch.norm(n_c[:, 1:, :, :] - n_c[:, :-1, :, :], dim=-1, keepdim=True)
                curv_c = F.pad(dn_dx, (0, 0, 0, 1, 0, 0)) + F.pad(dn_dy, (0, 0, 0, 0, 0, 1))

                dd_dx = torch.abs(d_c[:, :, 1:, :] - d_c[:, :, :-1, :])
                dd_dy = torch.abs(d_c[:, 1:, :, :] - d_c[:, :-1, :, :])
                grad_d_c = F.pad(dd_dx, (0, 0, 0, 1, 0, 0)) + F.pad(dd_dy, (0, 0, 0, 0, 0, 1))

                # On the 128x128 canonical grid, step-edge suppression exp(-8 * |grad_c d|) is scale-invariant
                interior_curv_c = curv_c * torch.exp(-grad_d_c * 8.0)
                curv_smooth_c = F.avg_pool2d(
                    interior_curv_c.permute(0, 3, 1, 2).contiguous(),
                    kernel_size=5,
                    stride=1,
                    padding=2,
                )
                curv_full = F.interpolate(
                    curv_smooth_c,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()[0, y0:y1, x0:x1, 0]

                score = curv_full * sub_fg.float()
                fg_scores = score[sub_fg]
                if fg_scores.numel() > 0:
                    p90_score = float(_safe_quantile(fg_scores, 0.90).item())
                    p50_score = float(_safe_quantile(fg_scores, 0.50).item())
                    # Confidence Gate: requires genuine 3D surface curvature (human face or smooth 3D android head)
                    if p90_score >= 0.015:
                        if p90_score >= 1.25 * (p50_score + 1e-5):
                            q_cut = _safe_quantile(fg_scores, 0.75)
                            salient_mask = sub_fg & (score >= q_cut)
                        else:
                            # Smooth 3D head/ellipsoid with uniform curvature across its crown
                            dn_sub = dn[y0:y1, x0:x1]
                            d_cut = _safe_quantile(dn_sub[sub_fg], 0.40)
                            q_cut = _safe_quantile(fg_scores, 0.45)
                            salient_mask = sub_fg & (score >= q_cut) & (dn_sub >= d_cut)
                        if int(salient_mask.sum().item()) >= 4:
                            pts = ps[y0:y1, x0:x1][salient_mask]
                            return torch.median(pts, dim=0).values.view(1, 3)
            elif not is_uniform_plane:
                # Depth-only fallback when normals are unavailable: require localized depth protrusion
                dn_sub = dn[y0:y1, x0:x1]
                if float(dn_sub[sub_fg].max().item() - dn_sub.min().item()) >= 0.12:
                    d_cut = _safe_quantile(dn_sub[sub_fg], 0.60)
                    salient_mask = sub_fg & (dn_sub >= d_cut)
                    if int(salient_mask.sum().item()) >= 4:
                        pts = ps[y0:y1, x0:x1][salient_mask]
                        return torch.median(pts, dim=0).values.view(1, 3)
        # Confidence Fallback: if upper portrait zone is empty, flat, or low-confidence, use Subject Centroid
        res_c = _region_fg_median(0.10, 0.90, 0.15, 0.85)
        return res_c if res_c is not None else default_center

    if aim_mode == "Center":
        res_m = _region_fg_median(0.34, 0.66, 0.20, 0.80)
        return res_m if res_m is not None else default_center

    if aim_mode in ("Lower", "Lower Third"):
        res_l = _region_fg_median(0.64, 0.92, 0.20, 0.80)
        return res_l if res_l is not None else torch.tensor([[0.0, -0.42, -2.0]], device=device, dtype=dtype)

    return default_center


def compute_light_shading(
    normals: torch.Tensor,
    light_vec: torch.Tensor,
    light_color: Tuple[float, float, float],
    intensity: float,
    modifier_name: str,
    depth: Optional[torch.Tensor] = None,
    distance: float = 1.8,
    return_separate: bool = False,
    shadow_samples: int = 8,
    shadow_steps: int = 16,
    target_pos: Optional[torch.Tensor] = None,
    vertical_fov_deg: float = 40.0,
    aim_target: str = "Upper",
    depth_relief_scale: float = 0.30,
    precomputed_surface: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None,
) -> torch.Tensor:
    """
    Computes diffuse irradiance and specular reflection using a physically based
    geometric area-emitter model.
    - Aiming: Resolves `aim_point` once and uses `p_light = aim_point + distance * stand_direction`
      and `beam_axis = normalize(aim_point - p_light)` across diffuse, specular, and 3D shadows.
    - Diffuse: Multi-sample area emitter integration with pure Lambertian `max(N·L, 0)`,
      smooth FWHM beam profile (`evaluate_beam_profile`), and 1/r^2 falloff.
    - Shadow: Metric camera-space 3D shadow raymarching with perspective projection
      and multi-sample area aperture penumbras.
    - Specular: Analytical aperture footprint in reflection space.
    """
    preset = MODIFIER_PRESETS.get(modifier_name, MODIFIER_PRESETS["Octabox (Soft)"])
    device = normals.device
    dtype = normals.dtype

    # Unit stand direction from aim_point toward the light fixture
    l_vec_raw = light_vec.to(device=device, dtype=dtype).view(1, 1, 1, 3)
    stand_direction = l_vec_raw / (torch.norm(l_vec_raw, dim=-1, keepdim=True) + 1e-7)
    r_stand = max(float(distance), 0.3)

    # View vector (towards camera +Z)
    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3)
    n_dot_v = torch.clamp(torch.sum(normals * v_vec, dim=-1, keepdim=True), 0.0, 1.0)
    refl_rays = 2.0 * n_dot_v * normals - v_vec
    refl_rays = refl_rays / (torch.norm(refl_rays, dim=-1, keepdim=True) + 1e-7)

    if depth is not None:
        B, H, W, _ = normals.shape
        aspect = float(W) / float(H)
        if precomputed_surface is not None:
            p_surf, d_norm, d_split = precomputed_surface
        else:
            p_surf, d_norm, d_split = reconstruct_metric_surface(
                depth.to(device=device, dtype=dtype),
                H=H,
                W=W,
                vertical_fov_deg=float(vertical_fov_deg),
                depth_relief_scale=float(depth_relief_scale),
            )

        # Resolve unified 3D aim_point [1, 3] and place fixture relative to aim_point
        if target_pos is not None:
            aim_point_13 = target_pos.to(device=device, dtype=dtype).reshape(-1, 3)[:1]
        else:
            aim_point_13 = resolve_aim_target(
                p_surf=p_surf,
                d_norm=d_norm,
                normals=normals,
                aim_mode=aim_target,
                d_split=d_split,
                target_pos=None,
                device=device,
                dtype=dtype,
            )
        p_light = aim_point_13.view(1, 1, 1, 3) + r_stand * stand_direction
        beam_axis_vec = aim_point_13.view(1, 1, 1, 3) - p_light
        beam_axis_vec = beam_axis_vec / (torch.norm(beam_axis_vec, dim=-1, keepdim=True) + 1e-7)
        u_tangent, v_tangent, w_aim = generate_emitter_tangent_frame(-beam_axis_vec.view(3))

        # 1. DIFFUSE EMISSION: Vectorized Area-Light Integration (16 samples)
        num_samples = 16
        samples_2d = generate_emitter_samples_2d(preset, num_samples=num_samples).to(device=device, dtype=dtype)
        # [1, S, 1, 1, 3] sample points in camera space (broadcast-safe for arbitrary batch B >= 1)
        p_samples = (
            p_light.view(1, 1, 1, 1, 3)
            + samples_2d[:, 0].view(1, num_samples, 1, 1, 1) * u_tangent.view(1, 1, 1, 1, 3)
            + samples_2d[:, 1].view(1, num_samples, 1, 1, 1) * v_tangent.view(1, 1, 1, 1, 3)
        )

        # Vectorized [B, S, H, W, 3] broadcast
        v_to_samples = p_samples - p_surf.unsqueeze(1)  # [B, S, H, W, 3]
        dist_to_samples = torch.norm(v_to_samples, dim=-1, keepdim=True) + 1e-6
        dir_to_samples = v_to_samples / dist_to_samples

        # Pure Lambertian cosine at receiver max(N . L, 0) and emitter off-axis cosine (w_aim . -L)
        cos_p = torch.clamp(torch.sum(normals.unsqueeze(1) * dir_to_samples, dim=-1, keepdim=True), 0.0, 1.0)
        cos_e = torch.clamp(torch.sum(w_aim.view(1, 1, 1, 1, 3) * (-dir_to_samples), dim=-1, keepdim=True), 0.0, 1.0)

        # Smooth calibrated FWHM + penumbra skirt beam profile
        beam_gate = evaluate_beam_profile(cos_e, preset)

        # True physical 1 / r^2 falloff (referenced to 1.8m studio portrait stand)
        inv_sq = torch.clamp((1.8 / dist_to_samples) ** 2.0, 0.0, 8.0)

        # Average irradiance across samples -> [B, H, W, 1]
        irradiance_samples = cos_p * beam_gate * inv_sq
        raw_diffuse = torch.mean(irradiance_samples, dim=1)

        # 2. SHADOW / VISIBILITY: Metric Camera-Space 3D Area Shadow Raymarching
        aim_point = aim_point_13.expand(B, 3)
        cast_sh_map = screen_space_directional_shadow_metric(
            surface_xyz=p_surf,
            scene_depth=-p_surf[..., 2],
            light_pos=p_light.view(1, 3).expand(B, 3),
            light_aim=aim_point,
            modifier=modifier_name,
            vertical_fov_deg=float(vertical_fov_deg),
            aspect=aspect,
            num_emitter_samples=shadow_samples,
            ray_steps=shadow_steps,
            sample_chunk=4,
        )
        cast_sh = cast_sh_map.unsqueeze(-1)

        # 3. SPECULAR REFLECTION: Physical Aperture Footprint in Reflection Space
        specular = evaluate_fixture_aperture_footprint(
            refl_rays,
            p_surf,
            p_light,
            u_tangent,
            v_tangent,
            w_aim,
            preset,
            roughness=preset.get("specular_roughness", 0.18),
        )
        # Horizon gating using local surface-to-fixture vector to prevent silhouette leakage
        v_to_center = p_light - p_surf
        l_local_dir = v_to_center / (torch.norm(v_to_center, dim=-1, keepdim=True) + 1e-6)
        horizon_gate = torch.clamp(torch.sum(normals * l_local_dir, dim=-1, keepdim=True) * 10.0, 0.0, 1.0)
        specular = specular * horizon_gate * float(preset.get("specular_strength", 0.50)) * cast_sh

    else:
        # Analytical point approximation when depth is unavailable
        l_vec = stand_direction
        dist_val = max(float(distance), 0.3)
        dist_ratio = 1.8 / dist_val
        attenuation = float(torch.clamp(torch.tensor(dist_ratio) ** 2.0, min=0.0, max=4.0))
        cast_sh = 1.0

        n_dot_l = torch.clamp(torch.sum(normals * l_vec, dim=-1, keepdim=True), 0.0, 1.0)
        raw_diffuse = n_dot_l * attenuation

        h_vec = l_vec + v_vec
        h_vec = h_vec / (torch.norm(h_vec, dim=-1, keepdim=True) + 1e-7)
        n_dot_h = torch.clamp(torch.sum(normals * h_vec, dim=-1, keepdim=True), 0.0, 1.0)
        spec_power = preset.get("specular_power", 32.0)
        specular = torch.pow(n_dot_h, spec_power) * float(preset.get("specular_strength", 0.50)) * attenuation

    c_tensor = torch.tensor(light_color, device=device, dtype=dtype).view(1, 1, 1, 3)
    diffuse_rgb = raw_diffuse * float(intensity) * c_tensor * cast_sh
    specular_rgb = specular * float(intensity) * c_tensor

    if return_separate:
        return (diffuse_rgb, specular_rgb)
    return diffuse_rgb + specular_rgb


def draw_studio_radar_diagram(
    key_params: Dict[str, Any],
    fill_params: Dict[str, Any],
    rim_params: Dict[str, Any],
    width: int = 512,
    height: int = 256,
    aim_point: Optional[torch.Tensor] = None,
    aim_target: str = "Upper",
) -> torch.Tensor:
    """
    Renders an informative 2D photographic studio radar compass showing Key, Fill,
    and Rim angles relative to Camera and Subject, plus the active 3D Aim Target HUD.
    Returns: [1, height, width, 3] float tensor in [0, 1] range.
    """
    img = Image.new("RGB", (width, height), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)

    cx = width // 2
    cy = height // 2
    radius = min(cx, cy) - 28

    # Draw radar concentric circles and crosshairs
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(55, 55, 62), width=2)
    draw.ellipse([cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2], outline=(38, 38, 44), width=1)
    draw.line([cx - radius, cy, cx + radius, cy], fill=(42, 42, 48), width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=(42, 42, 48), width=1)

    # Diagonal guidelines (+/- 45 deg, +/- 135 deg)
    for angle in [45, 135, -45, -135]:
        rad = math.radians(angle)
        gx = cx + int(radius * math.sin(rad))
        gy = cy + int(radius * math.cos(rad))
        draw.line([cx, cy, gx, gy], fill=(32, 32, 36), width=1)

    # Subject center / aim target marker
    ax_val, ay_val, az_val = 0.0, 0.0, -2.0
    if aim_point is not None and aim_point.numel() >= 3:
        ap = aim_point.detach().cpu().reshape(-1)
        ax_val, ay_val, az_val = float(ap[0]), float(ap[1]), float(ap[2])

    aim_px = cx + int(max(-1.0, min(1.0, ax_val)) * (radius * 0.45))
    aim_py = cy + int(max(-1.0, min(1.0, (az_val + 2.0))) * (radius * 0.35))
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(90, 90, 98), outline=(65, 65, 72), width=1)
    # Active 3D Aim Target crosshair
    draw.ellipse([aim_px - 6, aim_py - 6, aim_px + 6, aim_py + 6], fill=(40, 235, 140), outline=(255, 255, 255), width=2)
    draw.line([aim_px - 10, aim_py, aim_px + 10, aim_py], fill=(40, 235, 140), width=1)
    draw.line([aim_px, aim_py - 10, aim_px, aim_py + 10], fill=(40, 235, 140), width=1)

    # Vertical Elevation Target Gauge (Right margin)
    gauge_x = width - 22
    g_top = 28
    g_bot = height - 28
    draw.rectangle([gauge_x - 4, g_top, gauge_x + 4, g_bot], fill=(28, 28, 34), outline=(70, 70, 80), width=1)
    y_norm_aim = max(0.0, min(1.0, 0.5 - (ay_val / 1.2)))
    gy_marker = int(g_top + y_norm_aim * (g_bot - g_top))
    draw.rectangle([gauge_x - 7, gy_marker - 3, gauge_x + 7, gy_marker + 3], fill=(40, 235, 140), outline=(255, 255, 255), width=1)
    draw.text((gauge_x - 24, g_top - 12), "AIM Y", fill=(150, 220, 175))

    # Top-left telemetry badge showing active aim mode & 3D coordinates
    short_mode = aim_target.split(" (")[0] if aim_target else "Upper"
    draw.text((8, 6), f"AIM [{short_mode}]: ({ax_val:+.2f}, {ay_val:+.2f}, {az_val:.2f}m)", fill=(130, 235, 165))

    # Bottom Camera marker (0°)
    cam_y = cy + radius + 14
    draw.rectangle([cx - 24, cam_y - 8, cx + 24, cam_y + 8], fill=(30, 30, 36), outline=(90, 90, 100), width=1)
    draw.text((cx - 19, cam_y - 6), "CAMERA", fill=(200, 200, 210))

    # Helper to draw light vector with distance-scaled radius
    def draw_vector(azimuth: float, elevation: float, color_rgb: Tuple[int, int, int], label: str, enabled: bool, distance: float = 1.8):
        if not enabled:
            return
        # Azimuth: 0° is bottom (camera), +45° is camera-right (screen right), +/-180° is top (backlight)
        ang_rad = math.radians(azimuth)
        dist_factor = min(max(float(distance) / 3.2, 0.32), 1.0)
        actual_r = int(radius * dist_factor)
        lx = cx + int(actual_r * math.sin(ang_rad))
        ly = cy + int(actual_r * math.cos(ang_rad))

        # Ray line aimed at active target marker
        draw.line([aim_px, aim_py, lx, ly], fill=color_rgb, width=3)
        # Light head node
        draw.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=color_rgb, outline=(255, 255, 255), width=2)
        # Degree and distance label
        tx = lx + (10 if math.sin(ang_rad) >= 0 else -68)
        ty = ly + (6 if math.cos(ang_rad) >= 0 else -14)
        draw.text((tx, ty), f"{label} {int(azimuth)}° ({distance:.1f}m)", fill=color_rgb)

    # Draw Key (Amber), Fill (Cyan), Rim (Magenta)
    draw_vector(key_params["azimuth"], key_params["elevation"], (255, 185, 45), "KEY", key_params.get("enabled", True), key_params.get("distance", 1.8))
    draw_vector(fill_params["azimuth"], fill_params["elevation"], (65, 205, 255), "FILL", fill_params.get("enabled", True), fill_params.get("distance", 2.5))
    draw_vector(rim_params["azimuth"], rim_params["elevation"], (245, 75, 175), "RIM", rim_params.get("enabled", True), rim_params.get("distance", 1.5))

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def render_studio_lighting(
    normals: Optional[torch.Tensor],
    depth: Optional[torch.Tensor],
    key_params: Dict[str, Any],
    fill_params: Dict[str, Any],
    rim_params: Dict[str, Any],
    ambient_intensity: float = 0.05,
    invert_normals_y: bool = False,
    normal_boost: float = 1.35,
    ao_anchoring: float = 0.60,
    preview_resolution: int = 512,
    return_passes: bool = False,
    invert_normals_x: bool = True,
    invert_depth: bool = False,
    target_pos: Optional[torch.Tensor] = None,
    vertical_fov_deg: float = 40.0,
    aim_target: str = "Upper",
    depth_relief_scale: float = 0.30,
) -> Tuple[torch.Tensor, ...]:
    """
    Renders the complete 3-point photographic studio lighting pass and dual-probe preview.
    Returns:
      relighting_map: [B, H, W, 3] in range [0, 1]
      probe_preview:  [1, H, W, 3] in range [0, 1] (Grey ball + Chrome ball + 2D Radar)
    """
    device = get_preferred_device() if normals is None else normals.device

    # Depth polarity: the shader assumes larger value = NEARER to camera (Depth Anything / ZoeDepth-style
    # "near = white"). Depth maps with near = black are flipped here, range-preserving, before any use
    # (back-projection, falloff, screen-space shadow and the subject mask all see the corrected map).
    # The flip normalizes to [0, 1] first so the absolute-valued subject mask (depth > 0.02) stays meaningful.
    if depth is not None and invert_depth:
        d_all = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        d_lo, d_hi = d_all.min(), d_all.max()
        if float(d_hi - d_lo) > 1e-6:
            depth = 1.0 - (d_all - d_lo) / (d_hi - d_lo)
        else:
            depth = d_all

    key_vec = azimuth_elevation_to_vector(key_params["azimuth"], key_params["elevation"])
    key_col = kelvin_to_rgb(key_params["kelvin"])

    fill_vec = azimuth_elevation_to_vector(fill_params["azimuth"], fill_params["elevation"])
    fill_col = kelvin_to_rgb(fill_params["kelvin"])

    rim_vec = azimuth_elevation_to_vector(rim_params["azimuth"], rim_params["elevation"])
    rim_col = kelvin_to_rgb(rim_params["kelvin"])

    is_procedural_fallback = (normals is None)
    asset_mask = None
    resolved_aim_point = target_pos
    total_diffuse = None
    total_specular = None
    subject_mask = None

    # 1. Relighting Map Computation: Procedural 3D calibration ellipsoid fallback when normals_optional is unconnected
    if normals is None:
        fh, fw = 1024, 1024
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, fh, device=device),
            torch.linspace(-1.0, 1.0, fw, device=device),
            indexing="ij",
        )
        r2 = (gx / 0.55) ** 2 + (gy / 0.65) ** 2
        inside = (r2 <= 1.0).float()
        nz = torch.sqrt(torch.clamp(1.0 - r2, min=0.0))
        nx = (gx / 0.55) * inside
        ny = (-gy / 0.65) * inside
        nz = nz * inside + (1.0 - inside)
        normals = torch.stack([(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, (nz + 1.0) * 0.5], dim=-1).unsqueeze(0)
        asset_mask = inside.unsqueeze(0).unsqueeze(-1)

    n = normals.clone()
    if n.shape[-1] != 3 and n.shape[1] == 3:
        n = n.permute(0, 2, 3, 1)

    norm_unpacked = n * 2.0 - 1.0
    if not is_procedural_fallback and invert_normals_x:
        norm_unpacked[..., 0] = -norm_unpacked[..., 0]
    if invert_normals_y:
        norm_unpacked[..., 1] = -norm_unpacked[..., 1]

    # Normal Amplification (Curvature Boost for shallow 2.5D diffusion drafts)
    if float(normal_boost) != 1.0:
        boost = max(float(normal_boost), 0.5)
        norm_unpacked[..., 0] = norm_unpacked[..., 0] * boost
        norm_unpacked[..., 1] = norm_unpacked[..., 1] * boost

    norm_len = torch.norm(norm_unpacked, dim=-1, keepdim=True) + 1e-7
    norm_unit = norm_unpacked / norm_len

    # Separate surface validity from relative depth magnitude:
    # In real scenes (e.g. DepthAnythingV2 normalized to [0, 1]), depth values < 0.02 belong to
    # the farthest background wall, not empty void. Pixels are excluded only when an explicit
    # alpha mask is present or when the normal map encodes genuinely invalid zero-length pixels (RGB norm <= 0.05).
    if asset_mask is not None:
        subject_mask = asset_mask
    else:
        raw_rgb_norm = torch.norm(n, dim=-1, keepdim=True)
        if bool((raw_rgb_norm > 0.05).any()):
            subject_mask = (raw_rgb_norm > 0.05).float()
        else:
            subject_mask = torch.ones_like(norm_unit[..., :1])

    total_diffuse = torch.zeros_like(norm_unit)
    total_specular = torch.zeros_like(norm_unit)

    precomputed_surf = None
    if depth is not None:
        B_n, H_n, W_n, _ = norm_unit.shape
        precomputed_surf = reconstruct_metric_surface(
            depth.to(device=norm_unit.device, dtype=norm_unit.dtype),
            H=H_n,
            W=W_n,
            vertical_fov_deg=float(vertical_fov_deg),
            depth_relief_scale=float(depth_relief_scale),
        )
        if resolved_aim_point is None:
            p_surf_tmp, d_norm_tmp, d_split_tmp = precomputed_surf
            resolved_aim_point = resolve_aim_target(
                p_surf=p_surf_tmp,
                d_norm=d_norm_tmp,
                normals=norm_unit,
                aim_mode=aim_target,
                d_split=d_split_tmp,
                target_pos=target_pos,
                device=norm_unit.device,
                dtype=norm_unit.dtype,
            )

    for p_dict, p_vec, p_col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p_dict.get("enabled", True):
            d_rgb, s_rgb = compute_light_shading(
                norm_unit, p_vec, p_col, p_dict["intensity"], p_dict["modifier"], depth,
                distance=p_dict.get("distance", 1.8),
                return_separate=True,
                target_pos=resolved_aim_point,
                vertical_fov_deg=float(vertical_fov_deg),
                aim_target=aim_target,
                depth_relief_scale=float(depth_relief_scale),
                precomputed_surface=precomputed_surf,
            )
            total_diffuse = total_diffuse + d_rgb
            total_specular = total_specular + s_rgb

    total_shading = total_diffuse + total_specular

    # Ambient Occlusion / Cavity Anchoring from Normal Field Divergence
    if float(ao_anchoring) > 0.0:
        cavity_ao = compute_cavity_ao_map(norm_unit, strength=float(ao_anchoring) * 2.5)
        total_shading = total_shading * cavity_ao

    # Composite subject with dark studio cyc backdrop
    bg_backdrop = torch.full_like(total_shading, max(float(ambient_intensity), 0.03))
    relighting_map = (total_shading + float(ambient_intensity)) * subject_mask + bg_backdrop * (1.0 - subject_mask)
    relighting_map = torch.clamp(relighting_map, 0.0, 1.0)

    # 2. Dual-Sphere Probe Preview Generation (Top: Balls, Bottom: Radar)
    probe_res = preview_resolution
    half_h = probe_res // 2
    half_w = probe_res // 2

    sphere_normals, sphere_mask = render_sphere_normals(half_h, device=device)

    # 3D sphere surface geometry in probe space (standard 25cm VFX calibration sphere)
    r_probe = 0.25
    p_sphere = sphere_normals * r_probe

    # Matte 18% Grey Ball with true 3D Euclidean distance falloff
    grey_shd = torch.zeros_like(sphere_normals)
    for p_dict, p_vec, p_col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p_dict.get("enabled", True):
            d_val = max(float(p_dict.get("distance", 1.8)), 0.3)
            lv_dir = p_vec.to(device=device, dtype=sphere_normals.dtype).view(1, 1, 1, 3)
            p_light_probe = d_val * lv_dir
            v_to_probe_light = p_light_probe - p_sphere
            dist_probe = torch.norm(v_to_probe_light, dim=-1, keepdim=True) + 1e-6
            l_probe_local = v_to_probe_light / dist_probe

            # Physical inverse-square falloff on the sphere surface referenced to 1.8m standard
            probe_dist_ratio = 1.8 / dist_probe
            probe_attenuation = torch.clamp(torch.pow(probe_dist_ratio, 1.5), min=0.10, max=2.8)

            n_dot_l_probe = torch.clamp(torch.sum(sphere_normals * l_probe_local, dim=-1, keepdim=True), min=0.0, max=1.0)
            mod_preset = MODIFIER_PRESETS.get(p_dict.get("modifier", "Octabox (Soft)"), MODIFIER_PRESETS["Octabox (Soft)"])
            wrap_p = min(mod_preset.get("wrap_factor", 0.0), 0.45)
            if wrap_p > 0.0:
                diff_p = torch.clamp((n_dot_l_probe + wrap_p) / (1.0 + wrap_p), min=0.0, max=1.0)
            else:
                diff_p = n_dot_l_probe

            col_t = torch.tensor(p_col, device=device, dtype=sphere_normals.dtype).view(1, 1, 1, 3)
            grey_shd = grey_shd + diff_p * probe_attenuation * float(p_dict["intensity"]) * col_t

    grey_ball = torch.clamp((grey_shd + ambient_intensity) * sphere_mask * 0.72, 0.0, 1.0)

    # Mirror Chrome Ball with distance-dependent specular spot size and intensity
    chrome_ball = torch.zeros_like(sphere_normals)
    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=sphere_normals.dtype).view(1, 1, 1, 3)
    n_dot_v = torch.sum(sphere_normals * v_vec, dim=-1, keepdim=True)
    refl_rays = 2.0 * n_dot_v * sphere_normals - v_vec
    refl_rays = refl_rays / (torch.norm(refl_rays, dim=-1, keepdim=True) + 1e-7)

    for p, vec, col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p.get("enabled", True):
            d_val = max(float(p.get("distance", 1.8)), 0.3)
            lv_dir = vec.to(device=device, dtype=sphere_normals.dtype).view(3)
            p_light_probe = d_val * lv_dir
            u_t, v_t, w_a = generate_emitter_tangent_frame(lv_dir)

            mod_preset = MODIFIER_PRESETS.get(p.get("modifier", "Octabox (Soft)"), MODIFIER_PRESETS["Octabox (Soft)"])
            # True photographic geometric aperture reflection on mirror chrome ball
            spec_spot = evaluate_fixture_aperture_footprint(
                refl_rays,
                p_sphere,
                p_light_probe,
                u_t,
                v_t,
                w_a,
                mod_preset,
                roughness=0.01,
            )
            col_t = torch.tensor(col, device=device, dtype=sphere_normals.dtype).view(1, 1, 1, 3)
            chrome_ball = chrome_ball + spec_spot * float(p["intensity"]) * col_t

    chrome_ball = torch.clamp((chrome_ball + ambient_intensity * 0.12) * sphere_mask, 0.0, 1.0)

    # Top half: Grey Ball & Chrome Ball side-by-side
    top_half = torch.cat([grey_ball, chrome_ball], dim=2)  # [1, half_h, probe_res, 3]

    # Bottom half: 2D Studio Radar Diagram
    bottom_half = draw_studio_radar_diagram(
        key_params,
        fill_params,
        rim_params,
        width=probe_res,
        height=half_h,
        aim_point=resolved_aim_point,
        aim_target=aim_target,
    )
    bottom_half = bottom_half.to(device=device)

    # Complete square probe preview card
    probe_preview = torch.cat([top_half, bottom_half], dim=1)  # [1, probe_res, probe_res, 3]

    if return_passes:
        diff_out = (total_diffuse * subject_mask).cpu() if total_diffuse is not None else relighting_map.cpu()
        spec_out = (total_specular * subject_mask).cpu() if total_specular is not None else torch.zeros_like(relighting_map).cpu()
        return relighting_map.cpu(), probe_preview.cpu(), diff_out, spec_out
    return relighting_map.cpu(), probe_preview.cpu()


def extract_photometric_carrier(flat_render: torch.Tensor, lit_render: torch.Tensor, epsilon: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates Linear RGB Photometric Ratio Division Map (L) and derives isolated Shadow/Highlight maps.
    
    Formula:
        L = (Lit_Render + eps) / (Flat_Render + eps)
        Shadow = max(0, 1 - L)
        Highlight = max(0, L - 1)
    """
    flat = flat_render.float()
    lit = lit_render.float()
    
    # Calculate lighting ratio map L
    ratio_map = (lit + epsilon) / (flat + epsilon)
    
    # Derive pure Shadow and Highlight maps
    shadow_map = torch.clamp(1.0 - ratio_map, min=0.0, max=1.0)
    highlight_map = torch.clamp(ratio_map - 1.0, min=0.0, max=1.0)
    
    return ratio_map, shadow_map, highlight_map

