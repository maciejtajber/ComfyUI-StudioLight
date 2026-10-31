"""
ComfyUI-StudioLight — Photometric Shader & Relighting Engine
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Calculates physical deferred lighting across surface normals (diffuse + specular),
exports 3D volumetric foreground & gated cast-shadow telemetry for scene-aware compositing,
and generates visual VFX probes with an integrated 2D studio radar compass.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, Optional, Tuple, Union
from .color_utils import kelvin_to_rgb, MODIFIER_PRESETS
from .metric_shadow import (
    _hammersley_2d,
    _regular_polygon_vertices,
    _sample_polygon_uniform,
    build_emitter_tangent_frame,
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
    """
    theta = math.radians(float(azimuth_deg))
    phi = math.radians(float(elevation_deg))

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

    curvature = (dnx_dx + dny_dy).permute(0, 2, 3, 1).contiguous()
    cavity = torch.clamp(1.0 + curvature * float(strength), min=0.35, max=1.0)
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
    """
    return build_emitter_tangent_frame(-light_dir)


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
    the geometric aperture footprint.
    """
    denom = torch.sum(refl_rays * w_aim.view(1, 1, 1, 3), dim=-1, keepdim=True)
    numer = torch.sum((p_light.view(1, 1, 1, 3) - surf_pos) * w_aim.view(1, 1, 1, 3), dim=-1, keepdim=True)
    # Invalid/back-facing rays cannot hit the emitting face. Use a finite
    # denominator before division: multiplying a NaN by a zero mask is unsafe.
    facing = denom < -0.01
    safe_denom = torch.where(facing, denom - 1e-6, torch.ones_like(denom))
    t = torch.where(facing, numer / safe_denom, torch.zeros_like(numer))
    valid_hit = (t > 0.0) & facing

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
    """
    beam_angle = float(preset.get("beam_angle_deg", 90.0))
    edge_softness = float(preset.get("edge_softness", 0.15))

    theta_half_deg = min(max(beam_angle * 0.5, 4.0), 82.0)
    cos_half = math.cos(math.radians(theta_half_deg))
    p_beam = math.log(0.5) / math.log(max(cos_half, 1e-4))

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
) -> Tuple[torch.Tensor, torch.Tensor, Union[float, torch.Tensor]]:
    """Construct an assumed-scale camera-space surface from near=1 relative depth.

    Positions are a controllable scene proxy, not measured metric geometry.
    Single-image split is a float; batches return Bx1x1x1 splits.
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

    if B > 1:
        p_list, dn_list, split_list = [], [], []
        for b_idx in range(B):
            p_b, dn_b, sp_b = reconstruct_metric_surface(
                d[b_idx : b_idx + 1],
                H=H,
                W=W,
                vertical_fov_deg=vertical_fov_deg,
                depth_relief_scale=depth_relief_scale,
            )
            p_list.append(p_b)
            dn_list.append(dn_b)
            split_list.append(sp_b)
        return torch.cat(p_list, dim=0), torch.cat(dn_list, dim=0), torch.tensor(split_list, device=device, dtype=dtype).view(B, 1, 1, 1)

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

        # Row-wise protrusion analysis: compares central spine against left/right room background at each row y
        dn_2d = d_norm[0, :, :, 0]
        row_bg_med = torch.median(dn_2d, dim=1, keepdim=True).values
        protrusion = dn_2d - row_bg_med

        cy0, cy1 = max(0, int(0.12 * H)), min(H, max(1, int(0.85 * H)))
        cx0, cx1 = max(0, int(0.25 * W)), min(W, max(1, int(0.75 * W)))
        sub_prot = protrusion[cy0:cy1, cx0:cx1]
        sub_dn = dn_2d[cy0:cy1, cx0:cx1]

        prot_mask = (sub_prot > 0.06) & (sub_dn > 0.10)
        if int(prot_mask.sum().item()) > max(32, int(0.01 * (cy1 - cy0) * (cx1 - cx0))):
            # Wide room / gallery / car shot: subject stands out in front of row background
            fg_vals = sub_dn[prot_mask]
            d_subj_val = float(torch.median(fg_vals).item())
            d_split_val = float(torch.clamp(_safe_quantile(fg_vals, 0.12) - 0.035, 0.11, 0.50).item())
        else:
            # Close-up portrait where subject fills the horizontal frame
            valid_c = sub_dn[sub_dn > 0.05].reshape(-1)
            if valid_c.numel() > 16:
                d_split_val = float(torch.clamp(_safe_quantile(valid_c, 0.20), 0.11, 0.50))
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


def compute_3d_subject_envelope(
    p_surf: torch.Tensor,
    d_norm: torch.Tensor,
    d_split: float,
    aim_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes a 3D volumetric foreground subject envelope `M_raw(x)` in [0, 1].
    Retained for backward-compatibility with external callers.
    """
    device = d_norm.device
    dtype = d_norm.dtype

    if aim_point is not None and aim_point.numel() >= 3:
        ap = aim_point.to(device=device, dtype=dtype).reshape(-1)
        aim_x = float(ap[0].item())
        aim_y = float(ap[1].item())
    else:
        aim_x, aim_y = 0.0, 0.20

    x_surf = p_surf[..., 0:1]
    y_surf = p_surf[..., 1:2]

    lower_factor = torch.clamp((aim_y - y_surf) / 0.85, 0.0, 1.0)
    corridor_half_width_m = 0.42 + 0.26 * lower_factor

    lat_excess_m = torch.clamp(torch.abs(x_surf - aim_x) - corridor_half_width_m, min=0.0)
    d_eff_split = float(d_split) + 0.28 * torch.pow(torch.clamp(lat_excess_m / 0.35, 0.0, 1.5), 2.0)

    raw_fg = torch.sigmoid((d_norm - d_eff_split) / 0.024)
    return torch.clamp(raw_fg, 0.0, 1.0)


def compute_3d_foreground_confidence(
    p_surf: torch.Tensor,
    d_norm: torch.Tensor,
    normals: torch.Tensor,
    aim_point: torch.Tensor,
    d_split: float,
) -> torch.Tensor:
    """
    Computes a 3D volumetric foreground subject confidence mask `fg_confidence` [B, H, W, 1] in [0, 1]
    that eliminates floor planes and side-wall columns from `M_fg`:
      1. Row-Protrusion & Z-Depth Gate: Compares pixel depth against both `d_split` and the row-wise
         outer-margin background depth so foreground floors and side columns at similar Z-depth are rejected.
      2. Floor/Ceiling Normal Exclusion: Pixels in the lower frame with upward-pointing surface normals
         (`ny > 0.58`) are physical floor planes and are smoothly excluded from `M_fg`.
      3. 3D Lateral Spine Envelope: Attenuates distant side walls beyond expected subject width around `aim_point.x`.
    """
    B, H, W, _ = d_norm.shape
    device, dtype = d_norm.device, d_norm.dtype

    # 1. Base Z-depth sigmoid
    z_gate = torch.sigmoid((d_norm - d_split) / 0.025)

    # 2. Row-wise outer-margin background comparison (catches floor ramps & side columns that span to frame edges)
    margin_w = max(4, int(0.12 * W))
    left_margin = d_norm[:, :, :margin_w, :]
    right_margin = d_norm[:, :, W - margin_w :, :]
    border_ref = 0.5 * (left_margin.mean(dim=2, keepdim=True) + right_margin.mean(dim=2, keepdim=True))
    border_diff = d_norm - border_ref

    # If the left/right frame borders at row y are nearly as close as d_split (e.g. floor or side walls),
    # require the subject pixel to protrude in front of the row's border depth.
    border_is_close = torch.sigmoid((border_ref - (d_split - 0.08)) / 0.03)
    protrusion_gate = torch.sigmoid((border_diff - 0.045) / 0.020)
    row_aware_gate = z_gate * ((1.0 - border_is_close) + border_is_close * protrusion_gate)

    # 3. Floor & Ceiling Normal Gate:
    # Upward-pointing normals (ny > 0.55) in the lower half of the frame are horizontal floors, not the standing subject.
    y_norm = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype).view(1, H, 1, 1)
    ny = normals[..., 1:2]
    is_lower_floor = torch.clamp((y_norm - 0.48) / 0.15, 0.0, 1.0) * torch.clamp((ny - 0.52) / 0.20, 0.0, 1.0)
    is_upper_ceiling = torch.clamp((0.25 - y_norm) / 0.15, 0.0, 1.0) * torch.clamp((-ny - 0.58) / 0.20, 0.0, 1.0)
    plane_gate = (1.0 - is_lower_floor) * (1.0 - is_upper_ceiling)

    # 4. 3D Lateral Spine Envelope around aim_point.x
    # For wide shots (where central subject occupies <30% of width), bound lateral X distance from aim_point
    fg_width_frac = float((row_aware_gate > 0.5).float().mean().item())
    if fg_width_frac < 0.28:
        ap = aim_point.reshape(-1, 3)
        ax = ap[:, 0].view(-1, 1, 1, 1)
        dx_m = torch.abs(p_surf[..., 0:1] - ax)
        lateral_gate = torch.sigmoid((0.55 - dx_m) / 0.08)
    else:
        lateral_gate = 1.0

    return torch.clamp(row_aware_gate * plane_gate * lateral_gate, 0.0, 1.0)


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

    Dynamic Head-Top Tracking for Wide Shots:
    -----------------------------------------
    Instead of searching a fixed `0.05*H .. 0.30*H` window (which misses the head in wide gallery shots
    where the subject's head begins at `y = 0.31*H`), locates the topmost row `y_top` of the protruding
    foreground subject spine and anchors the `"Upper"` search window dynamically at `[y_top, y_top + 0.22*H]`.
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

    # Isolate true subject spine from side walls and floor
    margin_w = max(4, int(0.12 * W))
    left_m = dn[:, :margin_w].mean(dim=1, keepdim=True)
    right_m = dn[:, -margin_w:].mean(dim=1, keepdim=True)
    row_bg = torch.maximum(left_m, right_m)
    protrudes = (dn - row_bg) > 0.05
    fg_mask = (dn >= d_split) & (dn > 0.10)
    if normals is not None and normals.shape[1] == H and normals.shape[2] == W:
        fg_mask = fg_mask & (normals[0, :, :, 1] < 0.58)  # Exclude upward floor plane

    cx0_s, cx1_s = max(0, int(0.22 * W)), min(W, max(1, int(0.78 * W)))
    spine_mask = fg_mask[:, cx0_s:cx1_s] & protrudes[:, cx0_s:cx1_s]
    if int(spine_mask.sum().item()) > max(24, int(0.005 * H * W)):
        active_fg = fg_mask & protrudes
    else:
        active_fg = fg_mask

    # Locate vertical bounding span [y_subj_top, y_subj_bot] of the subject inside the central corridor
    row_counts = active_fg[:, cx0_s:cx1_s].sum(dim=1)
    valid_rows = torch.nonzero(row_counts >= max(3, int(0.015 * W)), as_tuple=False).reshape(-1)
    if valid_rows.numel() >= 4:
        y_subj_top = int(valid_rows[0].item())
        y_subj_bot = int(valid_rows[-1].item())
    else:
        y_subj_top = int(0.05 * H)
        y_subj_bot = int(0.90 * H)
    subj_h = max(int(0.15 * H), y_subj_bot - y_subj_top)

    def _region_fg_median(y0_px: int, y1_px: int, x0_f: float, x1_f: float) -> Optional[torch.Tensor]:
        y0, y1 = max(0, y0_px), min(H, max(y0_px + 1, y1_px))
        x0, x1 = max(0, int(x0_f * W)), min(W, max(1, int(x1_f * W)))
        sub_fg = active_fg[y0:y1, x0:x1]
        if int(sub_fg.sum().item()) < 8:
            sub_fg = fg_mask[y0:y1, x0:x1]
        if int(sub_fg.sum().item()) < 8:
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
        # Multi-scale head window: check primary portrait zone [0.05*H, 0.32*H] first;
        # if head is located lower in frame (wide gallery shots), dynamically shift to [y_subj_top, y_subj_top + 0.28*subj_h]
        candidate_windows = []
        candidate_windows.append((max(0, int(0.05 * H)), min(H, max(1, int(0.32 * H)))))
        if y_subj_top > int(0.20 * H):
            candidate_windows.append((max(0, y_subj_top), min(H, max(y_subj_top + 8, y_subj_top + int(0.28 * subj_h)))))

        x0, x1 = max(0, int(0.22 * W)), min(W, max(1, int(0.78 * W)))

        for (y0, y1) in candidate_windows:
            sub_fg = fg_mask[y0:y1, x0:x1]
            min_px = max(6, int(0.008 * (y1 - y0) * (x1 - x0)))
            is_uniform_plane = float(fg_mask.float().mean().item()) > 0.92
            if int(sub_fg.sum().item()) >= min_px:
                if normals is not None and normals.shape[1] == H and normals.shape[2] == W:
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
                        if p90_score >= 0.015:
                            if p90_score >= 1.25 * (p50_score + 1e-5):
                                q_cut = _safe_quantile(fg_scores, 0.75)
                                salient_mask = sub_fg & (score >= q_cut)
                            else:
                                dn_sub = dn[y0:y1, x0:x1]
                                d_cut = _safe_quantile(dn_sub[sub_fg], 0.40)
                                q_cut = _safe_quantile(fg_scores, 0.45)
                                salient_mask = sub_fg & (score >= q_cut) & (dn_sub >= d_cut)
                            if int(salient_mask.sum().item()) >= 4:
                                pts = ps[y0:y1, x0:x1][salient_mask]
                                return torch.median(pts, dim=0).values.view(1, 3)
                elif not is_uniform_plane:
                    dn_sub = dn[y0:y1, x0:x1]
                    if float(dn_sub[sub_fg].max().item() - dn_sub.min().item()) >= 0.10:
                        d_cut = _safe_quantile(dn_sub[sub_fg], 0.55)
                        salient_mask = sub_fg & (dn_sub >= d_cut)
                        if int(salient_mask.sum().item()) >= 4:
                            pts = ps[y0:y1, x0:x1][salient_mask]
                            return torch.median(pts, dim=0).values.view(1, 3)

        # Fallback if no salient curvature in candidate windows
        y0_dyn = max(0, y_subj_top)
        y1_dyn = min(H, max(y0_dyn + 8, y_subj_top + int(0.28 * subj_h)))
        res_head = _region_fg_median(y0_dyn, y1_dyn, 0.22, 0.78)
        if res_head is not None:
            return res_head
        res_c = _region_fg_median(y_subj_top, y_subj_bot, 0.20, 0.80)
        return res_c if res_c is not None else default_center

    if aim_mode == "Center":
        y0_m = y_subj_top + int(0.28 * subj_h)
        y1_m = y_subj_top + int(0.68 * subj_h)
        res_m = _region_fg_median(y0_m, y1_m, 0.20, 0.80)
        return res_m if res_m is not None else default_center

    if aim_mode in ("Lower", "Lower Third"):
        y0_l = y_subj_top + int(0.62 * subj_h)
        y1_l = y_subj_bot
        res_l = _region_fg_median(y0_l, y1_l, 0.20, 0.80)
        return res_l if res_l is not None else torch.tensor([[0.0, -0.42, -2.0]], device=device, dtype=dtype)

    return default_center


def integrate_area_diffuse(normals, surface_xyz, light_pos, aim_point, preset, num_samples=16, return_lambertian=False):
    """Normalized area-emitter irradiance used by scene and calibration probe."""
    device, dtype = normals.device, normals.dtype
    p_light = light_pos.reshape(1, 1, 1, 3)
    axis = aim_point.reshape(1, 1, 1, 3) - p_light
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-7)
    u, v, forward = generate_emitter_tangent_frame(-axis.view(3))
    samples = generate_emitter_samples_2d(preset, num_samples).to(device=device, dtype=dtype)
    points = (
        p_light.view(1, 1, 1, 1, 3)
        + samples[:, 0].view(1, num_samples, 1, 1, 1) * u.view(1, 1, 1, 1, 3)
        + samples[:, 1].view(1, num_samples, 1, 1, 1) * v.view(1, 1, 1, 1, 3)
    )
    offset = points - surface_xyz.unsqueeze(1)
    distance = torch.norm(offset, dim=-1, keepdim=True) + 1e-6
    direction = offset / distance
    n_dot_l = torch.sum(normals.unsqueeze(1) * direction, dim=-1, keepdim=True)
    wrap = 0.22
    receiver_cos = torch.pow(torch.clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0), 1.35)
    emitter_cos = torch.clamp(torch.sum(forward.view(1, 1, 1, 1, 3) * -direction, dim=-1, keepdim=True), 0.0, 1.0)
    beam = evaluate_beam_profile(emitter_cos, preset)
    legacy = torch.mean(receiver_cos * beam * (1.8 / distance) ** 2.0, dim=1)
    if return_lambertian:
        strict = torch.mean(n_dot_l.clamp(0, 1) * beam * (1.8 / distance) ** 2.0, dim=1)
        return legacy, strict
    return legacy


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
    return_shadow_telemetry: bool = False,
) -> Any:
    """
    Computes diffuse irradiance and specular reflection using a physically based
    geometric area-emitter model.
    """
    preset = MODIFIER_PRESETS.get(modifier_name, MODIFIER_PRESETS["Octabox (Soft)"])
    device = normals.device
    dtype = normals.dtype

    l_vec_raw = light_vec.to(device=device, dtype=dtype).view(1, 1, 1, 3)
    stand_direction = l_vec_raw / (torch.norm(l_vec_raw, dim=-1, keepdim=True) + 1e-7)
    r_stand = max(float(distance), 0.3)

    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3)
    n_dot_v = torch.clamp(torch.sum(normals * v_vec, dim=-1, keepdim=True), 0.0, 1.0)
    refl_rays = 2.0 * n_dot_v * normals - v_vec
    refl_rays = refl_rays / (torch.norm(refl_rays, dim=-1, keepdim=True) + 1e-7)

    if depth is not None and depth.shape[0] not in (1, normals.shape[0]):
        raise ValueError(
            f"StudioLight shading: depth batch ({depth.shape[0]}) must be 1 or match normals batch ({normals.shape[0]})."
        )
    if depth is not None and normals.shape[0] > 1:
        B_multi = normals.shape[0]
        diff_list, spec_list, sum_list, unocc_list, block_list, vol_list = [], [], [], [], [], []
        for b_idx in range(B_multi):
            n_b = normals[b_idx : b_idx + 1]
            d_b = depth[b_idx : b_idx + 1] if depth.shape[0] == B_multi else depth
            t_b = (
                target_pos[b_idx : b_idx + 1]
                if (target_pos is not None and target_pos.ndim >= 2 and target_pos.shape[0] == B_multi)
                else target_pos
            )
            res_b = compute_light_shading(
                n_b,
                light_vec,
                light_color,
                intensity,
                modifier_name,
                d_b,
                distance=distance,
                return_separate=return_separate,
                shadow_samples=shadow_samples,
                shadow_steps=shadow_steps,
                target_pos=t_b,
                vertical_fov_deg=vertical_fov_deg,
                aim_target=aim_target,
                depth_relief_scale=depth_relief_scale,
                precomputed_surface=None,
                return_shadow_telemetry=return_shadow_telemetry,
            )
            if return_separate and return_shadow_telemetry:
                diff_list.append(res_b[0])
                spec_list.append(res_b[1])
                unocc_list.append(res_b[2])
                block_list.append(res_b[3])
                vol_list.append(res_b[4])
            elif return_separate:
                diff_list.append(res_b[0])
                spec_list.append(res_b[1])
            else:
                sum_list.append(res_b)
        if return_separate and return_shadow_telemetry:
            return (
                torch.cat(diff_list, dim=0),
                torch.cat(spec_list, dim=0),
                torch.cat(unocc_list, dim=0),
                torch.cat(block_list, dim=0),
                torch.cat(vol_list, dim=0),
            )
        if return_separate:
            return (torch.cat(diff_list, dim=0), torch.cat(spec_list, dim=0))
        return torch.cat(sum_list, dim=0)

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

        v_vec = F.normalize(-p_surf, dim=-1, eps=1e-7)
        n_dot_v = torch.clamp(torch.sum(normals * v_vec, dim=-1, keepdim=True), 0.0, 1.0)
        refl_rays = F.normalize(2.0 * n_dot_v * normals - v_vec, dim=-1, eps=1e-7)

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

        raw_diffuse, strict_diffuse = integrate_area_diffuse(
            normals, p_surf, p_light, aim_point_13, preset, return_lambertian=True)

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
        cast_sh_raw = cast_sh_map.unsqueeze(-1)
        cast_bchw = cast_sh_raw.permute(0, 3, 1, 2).contiguous()
        cast_pad = F.pad(cast_bchw, (3, 3, 3, 3), mode="replicate")
        cast_blur = F.avg_pool2d(F.avg_pool2d(cast_pad, kernel_size=(1, 7), stride=1), kernel_size=(7, 1), stride=1)
        cast_sh = cast_blur.permute(0, 2, 3, 1).contiguous()

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
        v_to_center = p_light - p_surf
        l_local_dir = v_to_center / (torch.norm(v_to_center, dim=-1, keepdim=True) + 1e-6)
        horizon_gate = torch.clamp(torch.sum(normals * l_local_dir, dim=-1, keepdim=True) * 10.0, 0.0, 1.0)
        specular = specular * horizon_gate * float(preset.get("specular_strength", 0.50)) * cast_sh

        unocc_irradiance = raw_diffuse * float(intensity)
        blocked_irradiance = unocc_irradiance * (1.0 - cast_sh)

        # 2c. 3D Volumetric Tyndall Beam Field (for Fog / Smoke / Disco Haze):
        dist_from_aim = torch.norm(p_surf - aim_point_13.view(1, 1, 1, 3), dim=-1, keepdim=True)
        dir_to_light = -l_local_dir
        emitter_cos_center = torch.clamp(torch.sum((-beam_axis_vec) * (-dir_to_light), dim=-1, keepdim=True), 0.0, 1.0)
        beam_gate = evaluate_beam_profile(emitter_cos_center, preset)
        vol_reach = torch.exp(-torch.clamp(dist_from_aim - 1.2, min=0.0) / 2.8)
        c_tensor = torch.tensor(light_color, device=device, dtype=dtype).view(1, 1, 1, 3)
        vol_beam = cast_sh * beam_gate * vol_reach * (float(intensity) * 0.45) * c_tensor
        diffuse_cast_sh = 0.22 + 0.78 * cast_sh
    else:
        l_vec = stand_direction
        dist_val = max(float(distance), 0.3)
        dist_ratio = 1.8 / dist_val
        attenuation = dist_ratio ** 2.0
        cast_sh = 1.0
        diffuse_cast_sh = 1.0

        n_dot_l = torch.sum(normals * l_vec, dim=-1, keepdim=True)
        wrap = 0.22
        raw_diffuse = torch.pow(torch.clamp((n_dot_l + wrap) / (1.0 + wrap), 0.0, 1.0), 1.35) * attenuation
        strict_diffuse = n_dot_l.clamp(0, 1) * attenuation

        h_vec = l_vec + v_vec
        h_vec = h_vec / (torch.norm(h_vec, dim=-1, keepdim=True) + 1e-7)
        n_dot_h = torch.clamp(torch.sum(normals * h_vec, dim=-1, keepdim=True), 0.0, 1.0)
        spec_power = preset.get("specular_power", 32.0)
        specular = torch.pow(n_dot_h, spec_power) * float(preset.get("specular_strength", 0.50)) * attenuation

        unocc_irradiance = raw_diffuse * float(intensity)
        blocked_irradiance = torch.zeros_like(unocc_irradiance)
        c_tensor = torch.tensor(light_color, device=device, dtype=dtype).view(1, 1, 1, 3)
        vol_beam = torch.zeros_like(raw_diffuse * float(intensity) * c_tensor)

    diffuse_rgb = raw_diffuse * float(intensity) * c_tensor * diffuse_cast_sh
    specular_rgb = specular * float(intensity) * c_tensor
    # New contract: Lambertian cosine and actual visibility, without the
    # legacy 0.22 diffuse-wrap or 22% shadow-side artistic floor.
    diffuse_rgb._studiolight_surface_diffuse = strict_diffuse * float(intensity) * c_tensor * cast_sh

    if return_separate and return_shadow_telemetry:
        return (diffuse_rgb, specular_rgb, unocc_irradiance, blocked_irradiance, vol_beam)
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
    """
    img = Image.new("RGB", (width, height), color=(18, 18, 20))
    draw = ImageDraw.Draw(img)

    cx = width // 2
    cy = height // 2
    radius = min(cx, cy) - 28

    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(55, 55, 62), width=2)
    draw.ellipse([cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2], outline=(38, 38, 44), width=1)
    draw.line([cx - radius, cy, cx + radius, cy], fill=(42, 42, 48), width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=(42, 42, 48), width=1)

    for angle in [45, 135, -45, -135]:
        rad = math.radians(angle)
        gx = cx + int(radius * math.sin(rad))
        gy = cy + int(radius * math.cos(rad))
        draw.line([cx, cy, gx, gy], fill=(32, 32, 36), width=1)

    ax_val, ay_val, az_val = 0.0, 0.0, -2.0
    if aim_point is not None and aim_point.numel() >= 3:
        ap = aim_point.detach().cpu().reshape(-1)
        ax_val, ay_val, az_val = float(ap[0]), float(ap[1]), float(ap[2])

    aim_px = cx + int(max(-1.0, min(1.0, ax_val)) * (radius * 0.45))
    aim_py = cy + int(max(-1.0, min(1.0, (az_val + 2.0))) * (radius * 0.35))
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(90, 90, 98), outline=(65, 65, 72), width=1)
    draw.ellipse([aim_px - 6, aim_py - 6, aim_px + 6, aim_py + 6], fill=(40, 235, 140), outline=(255, 255, 255), width=2)
    draw.line([aim_px - 10, aim_py, aim_px + 10, aim_py], fill=(40, 235, 140), width=1)
    draw.line([aim_px, aim_py - 10, aim_px, aim_py + 10], fill=(40, 235, 140), width=1)

    gauge_x = width - 22
    g_top = 28
    g_bot = height - 28
    draw.rectangle([gauge_x - 4, g_top, gauge_x + 4, g_bot], fill=(28, 28, 34), outline=(70, 70, 80), width=1)
    y_norm_aim = max(0.0, min(1.0, 0.5 - (ay_val / 1.2)))
    gy_marker = int(g_top + y_norm_aim * (g_bot - g_top))
    draw.rectangle([gauge_x - 7, gy_marker - 3, gauge_x + 7, gy_marker + 3], fill=(40, 235, 140), outline=(255, 255, 255), width=1)
    draw.text((gauge_x - 24, g_top - 12), "AIM Y", fill=(150, 220, 175))

    short_mode = aim_target.split(" (")[0] if aim_target else "Upper"
    draw.text((8, 6), f"AIM [{short_mode}]: ({ax_val:+.2f}, {ay_val:+.2f}, {az_val:.2f}m)", fill=(130, 235, 165))

    cam_y = cy + radius + 14
    draw.rectangle([cx - 24, cam_y - 8, cx + 24, cam_y + 8], fill=(30, 30, 36), outline=(90, 90, 100), width=1)
    draw.text((cx - 19, cam_y - 6), "CAMERA", fill=(200, 200, 210))

    def draw_vector(azimuth: float, elevation: float, color_rgb: Tuple[int, int, int], label: str, enabled: bool, distance: float = 1.8):
        if not enabled:
            return
        ang_rad = math.radians(azimuth)
        dist_factor = min(max(float(distance) / 3.2, 0.32), 1.0)
        actual_r = int(radius * dist_factor)
        lx = cx + int(actual_r * math.sin(ang_rad))
        ly = cy + int(actual_r * math.cos(ang_rad))

        draw.line([aim_px, aim_py, lx, ly], fill=color_rgb, width=3)
        draw.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=color_rgb, outline=(255, 255, 255), width=2)
        tx = lx + (10 if math.sin(ang_rad) >= 0 else -68)
        ty = ly + (6 if math.cos(ang_rad) >= 0 else -14)
        draw.text((tx, ty), f"{label} {int(azimuth)}° ({distance:.1f}m)", fill=color_rgb)

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
    _render_probe: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """
    Renders the complete 3-point photographic studio lighting pass and dual-probe preview,
    attaching `_studiolight_telemetry` directly to `relighting_map` for downstream 3-Domain compositing.
    """
    if normals is not None and depth is not None:
        bn, bd = normals.shape[0], depth.shape[0]
        if bn < 1 or bd < 1 or (bn != bd and bn != 1 and bd != 1):
            raise ValueError(f"StudioLight: normals batch ({bn}) and depth batch ({bd}) must match or one must be 1.")
    elif (normals is not None and normals.shape[0] < 1) or (depth is not None and depth.shape[0] < 1):
        raise ValueError("StudioLight: image batches must contain at least one image.")
    b_norm = normals.shape[0] if normals is not None else 1
    b_depth = depth.shape[0] if depth is not None else 1
    B_batch = max(b_norm, b_depth)
    if B_batch > 1:
        relight_list, diff_list, spec_list = [], [], []
        telemetry_list = []
        probe_first = None
        for b_idx in range(B_batch):
            n_slice = (
                normals[b_idx : b_idx + 1]
                if (normals is not None and normals.shape[0] == B_batch)
                else normals
            )
            d_slice = (
                depth[b_idx : b_idx + 1]
                if (depth is not None and depth.shape[0] == B_batch)
                else depth
            )
            t_slice = (
                target_pos[b_idx : b_idx + 1]
                if (target_pos is not None and target_pos.ndim >= 2 and target_pos.shape[0] == B_batch)
                else target_pos
            )
            res_b = render_studio_lighting(
                normals=n_slice,
                depth=d_slice,
                key_params=key_params,
                fill_params=fill_params,
                rim_params=rim_params,
                ambient_intensity=ambient_intensity,
                invert_normals_y=invert_normals_y,
                normal_boost=normal_boost,
                ao_anchoring=ao_anchoring,
                preview_resolution=preview_resolution,
                return_passes=return_passes,
                invert_normals_x=invert_normals_x,
                invert_depth=invert_depth,
                target_pos=t_slice,
                vertical_fov_deg=vertical_fov_deg,
                aim_target=aim_target,
                depth_relief_scale=depth_relief_scale,
                _render_probe=(b_idx == 0 and _render_probe),
            )
            relight_list.append(res_b[0])
            tele_b = getattr(res_b[0], "_studiolight_telemetry", None)
            if tele_b is not None:
                telemetry_list.append(tele_b)
            if probe_first is None:
                probe_first = res_b[1]
            if return_passes:
                diff_list.append(res_b[2])
                spec_list.append(res_b[3])
        relight_cat = torch.cat(relight_list, dim=0)
        if len(telemetry_list) == B_batch:
            merged = {"schema_version": 2, "depth_convention": "relative_near_one",
                      "geometry_scale": "assumed", "ambient_intensity": float(ambient_intensity)}
            for name in ("d_norm", "fg_envelope", "fg_confidence", "shadow_occlusion", "volumetric_beam",
                         "direct_diffuse", "direct_specular", "ambient_rgb",
                         "surface_direct_diffuse", "surface_direct_specular", "surface_ambient_rgb"):
                if all(name in t for t in telemetry_list):
                    merged[name] = torch.cat([t[name] for t in telemetry_list], dim=0)
            if all("d_split" in t for t in telemetry_list):
                merged["d_split"] = torch.tensor([t["d_split"] for t in telemetry_list]).view(B_batch, 1, 1, 1)
            relight_cat._studiolight_telemetry = merged
        if return_passes:
            return relight_cat, probe_first, torch.cat(diff_list, dim=0), torch.cat(spec_list, dim=0)
        return relight_cat, probe_first

    device = get_preferred_device() if normals is None else normals.device

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

    if float(normal_boost) != 1.0:
        boost = max(float(normal_boost), 0.5)
        norm_unpacked[..., 0] = norm_unpacked[..., 0] * boost
        norm_unpacked[..., 1] = norm_unpacked[..., 1] * boost

    norm_len = torch.norm(norm_unpacked, dim=-1, keepdim=True) + 1e-7
    norm_unit = norm_unpacked / norm_len

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
    total_unocc_irr = torch.zeros_like(norm_unit[..., :1])
    total_blocked_irr = torch.zeros_like(norm_unit[..., :1])
    total_vol_beam = torch.zeros_like(norm_unit)

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

    total_surface_diffuse = torch.zeros_like(total_diffuse)
    for p_dict, p_vec, p_col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p_dict.get("enabled", True):
            d_rgb, s_rgb, unocc_irr, block_irr, vol_b = compute_light_shading(
                norm_unit,
                p_vec,
                p_col,
                p_dict["intensity"],
                p_dict["modifier"],
                depth,
                distance=p_dict.get("distance", 1.8),
                return_separate=True,
                target_pos=resolved_aim_point,
                vertical_fov_deg=float(vertical_fov_deg),
                aim_target=aim_target,
                depth_relief_scale=float(depth_relief_scale),
                precomputed_surface=precomputed_surf,
                return_shadow_telemetry=True,
            )
            total_diffuse = total_diffuse + d_rgb
            total_surface_diffuse = total_surface_diffuse + d_rgb._studiolight_surface_diffuse
            total_specular = total_specular + s_rgb
            total_unocc_irr = total_unocc_irr + unocc_irr
            total_blocked_irr = total_blocked_irr + block_irr
            total_vol_beam = total_vol_beam + vol_b

    surface_specular = total_specular
    if float(ao_anchoring) > 0.0:
        cavity_ao = compute_cavity_ao_map(norm_unit, strength=float(ao_anchoring) * 2.5)
        total_diffuse = total_diffuse * cavity_ao
        total_specular = total_specular * cavity_ao
    total_shading = total_diffuse + total_specular

    bg_backdrop = torch.full_like(total_shading, max(float(ambient_intensity), 0.03))
    relighting_map = (total_shading + float(ambient_intensity)) * subject_mask + bg_backdrop * (1.0 - subject_mask)
    relighting_map = torch.clamp(relighting_map, min=0.0)
    relighting_map_cpu = relighting_map.cpu()

    # Separated linear gains are available even when geometry is absent. These
    # fields exclude the specular pass and ambient from the bounce source.
    relighting_map_cpu._studiolight_telemetry = {
        "schema_version": 2,
        "depth_convention": "relative_near_one",
        "geometry_scale": "assumed",
        "direct_diffuse": (total_diffuse * subject_mask).detach().cpu(),
        "direct_specular": (total_specular * subject_mask).detach().cpu(),
        "ambient_rgb": (float(ambient_intensity) * subject_mask + bg_backdrop * (1 - subject_mask)).detach().cpu(),
        "ambient_intensity": float(ambient_intensity),
        # Full visible-surface passes for the explicit relighting contract.
        # Legacy subject-gated fields and IMAGE outputs above remain unchanged.
        "surface_direct_diffuse": total_surface_diffuse.detach().cpu(),
        "surface_direct_specular": surface_specular.detach().cpu(),
        "surface_ambient_rgb": torch.full_like(total_diffuse, float(ambient_intensity)).detach().cpu(),
    }

    # Attach 3D Volumetric Subject Envelope & Gated Cast-Shadow Telemetry
    if precomputed_surf is not None and resolved_aim_point is not None:
        p_surf_t, d_norm_t, d_split_t = precomputed_surf
        fg_conf_t = compute_3d_foreground_confidence(
            p_surf=p_surf_t,
            d_norm=d_norm_t,
            normals=norm_unit,
            aim_point=resolved_aim_point,
            d_split=d_split_t,
        )
        # Physical irradiance-gated cast shadow occlusion (immunizes self-shadowed walls & out-of-beam areas)
        reach_envelope = torch.clamp(total_unocc_irr / 0.08, 0.0, 1.0)
        shadow_occlusion = torch.clamp(
            (total_blocked_irr / (total_unocc_irr + 0.035)) * reach_envelope,
            0.0,
            1.0,
        )
        relighting_map_cpu._studiolight_telemetry.update({
            "d_norm": d_norm_t.detach().cpu(),
            "d_split": float(d_split_t),
            "fg_envelope": fg_conf_t.detach().cpu(),
            "fg_confidence": fg_conf_t.detach().cpu(),
            "shadow_occlusion": shadow_occlusion.detach().cpu(),
            "volumetric_beam": total_vol_beam.detach().cpu(),
            "ambient_intensity": float(ambient_intensity),
        })

    if not _render_probe:
        probe_preview = torch.empty((1, 1, 1, 3), dtype=torch.float32)
        if return_passes:
            diff_out = (total_diffuse * subject_mask).cpu()
            spec_out = (total_specular * subject_mask).cpu()
            return relighting_map_cpu, probe_preview, diff_out, spec_out
        return relighting_map_cpu, probe_preview

    # 2. Dual-Sphere Probe Preview Generation (Top: Balls, Bottom: Radar)
    probe_res = preview_resolution
    half_h = probe_res // 2

    sphere_normals, sphere_mask = render_sphere_normals(half_h, device=device)
    r_probe = 0.25
    p_sphere = sphere_normals * r_probe

    grey_shd = torch.zeros_like(sphere_normals)
    for p_dict, p_vec, p_col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p_dict.get("enabled", True):
            d_val = max(float(p_dict.get("distance", 1.8)), 0.3)
            light_pos = d_val * p_vec.to(device=device, dtype=sphere_normals.dtype).reshape(1, 3)
            aim = torch.zeros_like(light_pos)
            mod_preset = MODIFIER_PRESETS.get(p_dict.get("modifier", "Octabox (Soft)"), MODIFIER_PRESETS["Octabox (Soft)"])
            irradiance = integrate_area_diffuse(sphere_normals, p_sphere, light_pos, aim, mod_preset)
            col_t = torch.tensor(p_col, device=device, dtype=sphere_normals.dtype).view(1, 1, 1, 3)
            grey_shd = grey_shd + irradiance * float(p_dict["intensity"]) * col_t

    from .photometric_composite import luminance_hdr_tonemap, linear_to_srgb
    grey_linear = (grey_shd + ambient_intensity) * 0.18
    grey_ball = linear_to_srgb(luminance_hdr_tonemap(grey_linear)) * sphere_mask

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

    chrome_ball = linear_to_srgb(luminance_hdr_tonemap(chrome_ball + ambient_intensity)) * sphere_mask

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
        diff_out = (total_diffuse * subject_mask).cpu() if total_diffuse is not None else relighting_map_cpu
        spec_out = (total_specular * subject_mask).cpu() if total_specular is not None else torch.zeros_like(relighting_map_cpu)
        return relighting_map_cpu, probe_preview.cpu(), diff_out, spec_out
    return relighting_map_cpu, probe_preview.cpu()


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
