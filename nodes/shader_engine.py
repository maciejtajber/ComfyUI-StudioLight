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
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, Tuple
from .color_utils import kelvin_to_rgb, MODIFIER_PRESETS


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

    nx = normals[..., 0:1].permute(0, 3, 1, 2)
    ny = normals[..., 1:2].permute(0, 3, 1, 2)

    dnx_dx = F.conv2d(nx, kx, padding=1)
    dny_dy = F.conv2d(ny, ky, padding=1)

    # Negative divergence indicates a concave valley / seam
    curvature = (dnx_dx + dny_dy).permute(0, 2, 3, 1)

    # Crevices: curvature < 0 (normals point inward towards each other)
    cavity = torch.clamp(1.0 + curvature * float(strength), min=0.0, max=1.0)
    return cavity


def get_vogel_disk_samples(
    num_samples: int = 8,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Computes deterministic 2D Vogel disk sample offsets for area-light filtering."""
    if device is None:
        device = torch.device("cpu")
    golden_angle = 2.39996323  # radians
    indices = torch.arange(num_samples, device=device, dtype=dtype)
    r = torch.sqrt((indices + 0.5) / float(num_samples))
    theta = indices * golden_angle
    offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    return offsets  # [num_samples, 2]


def screen_space_directional_shadow(
    depth: torch.Tensor,
    light_vec: torch.Tensor,
    steps: int = 16,
    max_dist: float = 0.08,
    aperture_size: float = 0.0,
    dist_to_light: Optional[torch.Tensor] = None,
    soft_shadows: bool = True,
) -> torch.Tensor:
    """
    Calculates directional cast shadow from screen-space depth raymarching.
    Features Screen-Space Percentage-Closer Soft Shadows (PCSS) with Vogel disk sampling:
      - Sharp contact shadows near occluders (gap ~ 0)
      - Soft diffused penumbra on distant background surfaces (gap > 0) scaled by modifier aperture size
    depth: [B, H, W, 1]
    light_vec: [3]
    """
    B, H, W = depth.shape[0], depth.shape[1], depth.shape[2]
    device = depth.device
    dtype = depth.dtype

    lx = float(light_vec[0].item() if isinstance(light_vec, torch.Tensor) else light_vec[0])
    ly = -float(light_vec[1].item() if isinstance(light_vec, torch.Tensor) else light_vec[1])
    norm_2d = (lx**2 + ly**2)**0.5
    if norm_2d < 1e-4:
        return torch.ones_like(depth[..., :1])

    dx = (lx / norm_2d) * (max_dist / steps)
    dy = (ly / norm_2d) * (max_dist / steps)

    d_in = depth if depth.shape[-1] == 1 else depth[..., 0:1]
    d_perm = d_in.permute(0, 3, 1, 2)

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    base_grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    shadow = torch.ones_like(d_in)
    blocker_gap = torch.zeros_like(d_in)
    bias = 0.005

    for s in range(1, steps + 1):
        offset_x = dx * s * 2.0
        offset_y = dy * s * 2.0

        sample_grid = base_grid.clone()
        sample_grid[..., 0] += offset_x
        sample_grid[..., 1] += offset_y

        sampled_d = F.grid_sample(d_perm, sample_grid, mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1)

        depth_diff = sampled_d - d_in
        is_closer = depth_diff > bias
        weight = 1.0 - (s / (steps + 1))
        occlusion = torch.clamp((depth_diff - bias) * 15.0, 0.0, 1.0) * is_closer.float() * weight
        shadow = shadow - occlusion * 0.25

        # Accumulate blocker distance gap (receiver depth - blocker depth) for penumbra estimation
        blocker_gap = torch.maximum(blocker_gap, torch.clamp(depth_diff, min=0.0) * is_closer.float())

    # Attenuate screen shadow when light is grazing or behind the plane (lz <= 0)
    lz_val = float(light_vec[2].item() if isinstance(light_vec, torch.Tensor) else light_vec[2])
    front_factor = max(0.0, min(1.0, lz_val))
    shadow = 1.0 - (1.0 - shadow) * front_factor
    shadow = torch.clamp(shadow, 0.0, 1.0)

    # Percentage-Closer Soft Shadows (PCSS) via Vogel Disk Sampling
    # Contact shadows remain razor-sharp; distant cast shadows diffuse according to aperture size
    if soft_shadows and float(aperture_size) > 0.05:
        # Calculate light stand distance factor
        if dist_to_light is not None:
            l_dist = torch.clamp(dist_to_light, min=0.5, max=10.0)
        else:
            l_dist = 1.8

        # Penumbra radius in normalized screen coordinates:
        # penumbra = aperture * (receiver_depth - blocker_depth) / distance
        # Clamped to maximum 0.040 (~40 pixels in standard resolution) to prevent washed-out blur
        penumbra_radius = torch.clamp(
            (float(aperture_size) * blocker_gap / l_dist) * 0.12,
            min=0.0,
            max=0.040,
        )

        vogel_samples = get_vogel_disk_samples(8, device=device, dtype=dtype)
        sh_perm = shadow.permute(0, 3, 1, 2)
        sh_accum = torch.zeros_like(shadow)

        for i in range(8):
            vx, vy = vogel_samples[i, 0], vogel_samples[i, 1]
            tap_grid = base_grid.clone()
            tap_grid[..., 0] += vx * penumbra_radius[..., 0]
            tap_grid[..., 1] += vy * penumbra_radius[..., 0]
            sampled_sh = F.grid_sample(sh_perm, tap_grid, mode="bilinear", padding_mode="border", align_corners=True).permute(0, 2, 3, 1)
            sh_accum = sh_accum + sampled_sh

        shadow = sh_accum / 8.0

    return torch.clamp(shadow, 0.0, 1.0)


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


def compute_light_shading(
    normals: torch.Tensor,
    light_vec: torch.Tensor,
    light_color: Tuple[float, float, float],
    intensity: float,
    modifier_name: str,
    depth: Optional[torch.Tensor] = None,
    distance: float = 1.8,
    return_separate: bool = False,
) -> torch.Tensor:
    """
    Computes diffuse irradiance and specular reflection using a physically based
    geometric area-emitter model.
    - Diffuse: Multi-sample area emitter integration with true 1/r^2 falloff and
      single-layer geometric penumbra shadow visibility.
    - Specular: Analytical aperture footprint in reflection space.
    """
    preset = MODIFIER_PRESETS.get(modifier_name, MODIFIER_PRESETS["Octabox (Soft)"])
    device = normals.device
    dtype = normals.dtype
    l_vec_dir = light_vec.to(device=device, dtype=dtype).view(1, 1, 1, 3)
    u_tangent, v_tangent, w_aim = generate_emitter_tangent_frame(l_vec_dir.view(3))

    r_stand = max(float(distance), 0.3)
    p_light = torch.tensor([0.0, 0.0, -2.0], device=device, dtype=dtype).view(1, 1, 1, 3) + r_stand * l_vec_dir

    # View vector (towards camera +Z)
    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3)
    n_dot_v = torch.clamp(torch.sum(normals * v_vec, dim=-1, keepdim=True), 0.0, 1.0)
    refl_rays = 2.0 * n_dot_v * normals - v_vec
    refl_rays = refl_rays / (torch.norm(refl_rays, dim=-1, keepdim=True) + 1e-7)

    if depth is not None:
        d = depth.to(device=device, dtype=dtype)
        if d.shape[-1] != 1 and len(d.shape) == 4:
            d = d[..., 0:1]
        elif len(d.shape) == 3:
            d = d.unsqueeze(-1)
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # 3D Camera-Space Back-Projection
        B, H, W, _ = normals.shape
        if d.shape[1] != H or d.shape[2] != W:
            d_perm = d.permute(0, 3, 1, 2)
            d = F.interpolate(d_perm, size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

        aspect = float(W) / float(H)
        tan_half_fov = 0.364
        y_coords, x_coords = torch.meshgrid(
            torch.linspace(1.0, -1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij"
        )
        x_grid = x_coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)
        y_grid = y_coords.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1)

        d_flat = d.view(-1)
        if d_flat.numel() > 1000:
            d_min = torch.quantile(d_flat, 0.01)
            d_max = torch.quantile(d_flat, 0.99)
        else:
            d_min = torch.min(d)
            d_max = torch.max(d)
        d_range = d_max - d_min

        if d_range < 1e-4:
            d_norm = torch.ones_like(d)
            z_pos = torch.full_like(d, -2.0)
        else:
            d_norm = torch.clamp((d - d_min) / (d_range + 1e-6), 0.0, 1.0)
            z_pos = -(1.5 + (1.0 - d_norm) * 3.0)

        x_pos = x_grid * tan_half_fov * aspect * (-z_pos)
        y_pos = y_grid * tan_half_fov * (-z_pos)
        p_surf = torch.cat([x_pos, y_pos, z_pos], dim=-1)

        # Distance taper for background
        bg_taper = torch.clamp(d_norm / 0.20, 0.0, 1.0)
        bg_taper = bg_taper * bg_taper * (3.0 - 2.0 * bg_taper)

        # 1. DIFFUSE EMISSION: Vectorized Area-Light Integration (16 samples)
        num_samples = 16
        samples_2d = generate_emitter_samples_2d(preset, num_samples=num_samples).to(device=device, dtype=dtype)
        # [S, 1, 1, 3] sample points in camera space
        p_samples = p_light.view(1, 1, 1, 3) + samples_2d[:, 0].view(num_samples, 1, 1, 1) * u_tangent.view(1, 1, 1, 3) + samples_2d[:, 1].view(num_samples, 1, 1, 1) * v_tangent.view(1, 1, 1, 3)

        # Vectorized [S, H, W, 3] broadcast
        v_to_samples = p_samples - p_surf  # [S, H, W, 3]
        dist_to_samples = torch.norm(v_to_samples, dim=-1, keepdim=True) + 1e-6
        dir_to_samples = v_to_samples / dist_to_samples

        # Cosines at receiver (N . L) and emitter (w_aim . -L)
        cos_p = torch.clamp(torch.sum(normals * dir_to_samples, dim=-1, keepdim=True), 0.0, 1.0)
        cos_e = torch.clamp(torch.sum(w_aim.view(1, 1, 1, 3) * (-dir_to_samples), dim=-1, keepdim=True), 0.0, 1.0)

        # Beam angle profile cutoff
        beam_angle = float(preset.get("beam_angle_deg", 90.0))
        cos_cutoff = math.cos(math.radians(min(beam_angle * 0.5, 85.0)))
        beam_gate = torch.clamp((cos_e - cos_cutoff) / (1.0 - cos_cutoff + 1e-5), 0.0, 1.0)

        # True physical 1 / r^2 falloff (referenced to 1.8m studio portrait stand)
        inv_sq = torch.clamp((1.8 / dist_to_samples) ** 2.0, 0.0, 8.0) * bg_taper

        # Average irradiance across samples -> [1, H, W, 1]
        irradiance_samples = cos_p * beam_gate * inv_sq
        raw_diffuse = torch.mean(irradiance_samples, dim=0, keepdim=True)


        # 2. SHADOW / VISIBILITY: Single-layer penumbra from sample visibility
        # Sample representative points across fixture aperture (sharp binary ray-march, no PCSS post-soften)
        sh_accum = torch.zeros_like(d)
        vis_sample_indices = [0, num_samples // 4, num_samples // 2, (3 * num_samples) // 4]
        subj_center = torch.tensor([0.0, 0.0, -2.0], device=device, dtype=dtype)
        for s_idx in vis_sample_indices:
            l_sample_vec = p_samples[s_idx].view(3) - subj_center
            l_sample_vec = l_sample_vec / (torch.norm(l_sample_vec) + 1e-6)
            dist_s = dist_to_samples[s_idx]
            if dist_s.dim() == 3:
                dist_s = dist_s.unsqueeze(0)
            sample_sh = screen_space_directional_shadow(
                d,
                l_sample_vec,
                aperture_size=0.0,
                dist_to_light=dist_s,
                soft_shadows=False,
            )
            sh_accum += sample_sh
        cast_sh = sh_accum / float(len(vis_sample_indices))


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
        # Horizon gating to prevent silhouette leakage
        horizon_gate = torch.clamp(torch.sum(normals * l_vec_dir, dim=-1, keepdim=True) * 10.0, 0.0, 1.0)
        specular = specular * horizon_gate * float(preset.get("specular_strength", 0.50)) * bg_taper * cast_sh

    else:
        # Analytical point approximation when depth is unavailable
        l_vec = l_vec_dir
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
) -> torch.Tensor:
    """
    Renders an informative 2D photographic studio radar compass showing Key, Fill,
    and Rim angles relative to Camera and Subject.
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

    # Subject center marker
    draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=(220, 220, 225), outline=(100, 100, 105), width=2)

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

        # Ray line
        draw.line([cx, cy, lx, ly], fill=color_rgb, width=3)
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
    normal_boost: float = 2.0,
    ao_anchoring: float = 0.60,
    preview_resolution: int = 512,
    framing: str = "Portrait (Classical Bust)",
    detected_prompt: str = "",
    return_passes: bool = False,
    invert_normals_x: bool = True,
    invert_depth: bool = False,
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

    is_built_in_mannequin = (normals is None)

    # 1. Relighting Map Computation: Select framing mannequin, product asset, or volumetric cyc
    asset_mask = None
    if normals is None:
        effective_framing = framing
        if "Auto" in framing and detected_prompt:
            p_low = detected_prompt.lower()
            if any(w in p_low for w in ["car", "vehicle", "automobile", "porsche", "ferrari", "bmw", "mercedes", "motor", "hood", "windshield"]):
                effective_framing = "Automotive Cove (Car / Large Vehicle)"
            elif any(w in p_low for w in ["perfume", "bottle", "cosmetics", "serum", "watch", "jewelry", "flacon", "tabletop", "scent", "glass"]):
                effective_framing = "Tabletop / Luxury (Perfume Bottle / Small Product)"
            elif any(w in p_low for w in ["chair", "furniture", "sofa", "armchair", "lamp", "stool", "table", "product design"]):
                effective_framing = "Industrial Design (Chair / Medium Product)"
            elif any(w in p_low for w in ["full body", "full length", "standing", "gown", "dress", "shoes", "heels", "head to toe"]):
                effective_framing = "Full Body (Standing Lookbook)"
            elif any(w in p_low for w in ["american", "cowboy", "waist up", "hips", "mid-thigh", "jacket", "suit", "coat"]):
                effective_framing = "American Plan (3/4 Shot)"
            elif any(w in p_low for w in ["crowd", "group", "two models", "three models", "runway", "sitting", "lounging"]):
                effective_framing = "Volumetric Cyc (Crowds / Dynamic)"
            else:
                effective_framing = "Portrait (Classical Bust)"

        assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        if "Portrait" in effective_framing:
            asset_file = "mannequin_bust_normal.png"
        elif "American" in effective_framing:
            asset_file = "mannequin_medium_normal.png"
        elif "Full Body" in effective_framing:
            asset_file = "mannequin_full_normal.png"
        elif "Tabletop" in effective_framing or "Perfume" in effective_framing:
            asset_file = "product_perfume_normal.png"
        elif "Industrial" in effective_framing or "Chair" in effective_framing:
            asset_file = "product_chair_normal.png"
        elif "Automotive" in effective_framing or "Car" in effective_framing:
            asset_file = "product_car_normal.png"
        else:
            # Volumetric Cyc (rendered procedurally)
            asset_file = None

        if asset_file:
            asset_path = os.path.join(assets_dir, asset_file)
            if not os.path.exists(asset_path):
                asset_path = os.path.join(assets_dir, "studio_head_normal.png")
            if os.path.exists(asset_path):
                img_asset = Image.open(asset_path)
                if img_asset.mode == "RGBA":
                    arr_asset = np.array(img_asset).astype(np.float32) / 255.0
                    normals = torch.from_numpy(arr_asset[..., :3]).unsqueeze(0).to(device)
                    asset_mask = torch.from_numpy(arr_asset[..., 3:4]).unsqueeze(0).to(device)
                else:
                    img_asset = img_asset.convert("RGB")
                    arr_asset = np.array(img_asset).astype(np.float32) / 255.0
                    normals = torch.from_numpy(arr_asset).unsqueeze(0).to(device)
                    asset_mask = None

    if normals is not None:
        n = normals.clone()
        if n.shape[-1] != 3 and n.shape[1] == 3:
            n = n.permute(0, 2, 3, 1)

        norm_unpacked = n * 2.0 - 1.0
        if not is_built_in_mannequin and invert_normals_x:
            norm_unpacked[..., 0] = -norm_unpacked[..., 0]
        if invert_normals_y:
            norm_unpacked[..., 1] = -norm_unpacked[..., 1]

        # 1. Normal Amplification (Curvature Boost for shallow 2.5D diffusion drafts)
        if float(normal_boost) != 1.0:
            boost = max(float(normal_boost), 0.5)
            norm_unpacked[..., 0] = norm_unpacked[..., 0] * boost
            norm_unpacked[..., 1] = norm_unpacked[..., 1] * boost

        norm_len = torch.norm(norm_unpacked, dim=-1, keepdim=True) + 1e-7
        norm_unit = norm_unpacked / norm_len

        # Precise solid alpha mask (eliminates holes 100%)
        if asset_mask is not None:
            subject_mask = asset_mask
        elif depth is not None:
            d_mask = depth.to(device=norm_unit.device, dtype=norm_unit.dtype)
            if d_mask.shape[-1] != 1 and len(d_mask.shape) == 4:
                d_mask = d_mask[..., 0:1]
            subject_mask = (d_mask > 0.02).float()
        elif is_built_in_mannequin:
            corner_val = norm_unit[:, :6, :6, :].mean(dim=(1, 2), keepdim=True)
            diff_from_corner = torch.norm(norm_unit - corner_val, dim=-1, keepdim=True)
            subject_mask = (diff_from_corner > 0.04).float()
        else:
            # User-supplied normal map: do not assume corners are background
            subject_mask = torch.ones_like(norm_unit[..., :1])

        total_diffuse = torch.zeros_like(norm_unit)
        total_specular = torch.zeros_like(norm_unit)

        for p_dict, p_vec, p_col in [
            (key_params, key_vec, key_col),
            (fill_params, fill_vec, fill_col),
            (rim_params, rim_vec, rim_col),
        ]:
            if p_dict.get("enabled", True):
                d_rgb, s_rgb = compute_light_shading(
                    norm_unit, p_vec, p_col, p_dict["intensity"], p_dict["modifier"], depth,
                    distance=p_dict.get("distance", 1.8),
                    return_separate=True
                )
                total_diffuse = total_diffuse + d_rgb
                total_specular = total_specular + s_rgb

        total_shading = total_diffuse + total_specular

        # 2. Ambient Occlusion / Cavity Anchoring from Normal Field Divergence
        if float(ao_anchoring) > 0.0:
            cavity_ao = compute_cavity_ao_map(norm_unit, strength=float(ao_anchoring) * 2.5)
            total_shading = total_shading * cavity_ao

        # Composite subject with dark studio cyc backdrop
        bg_backdrop = torch.full_like(total_shading, max(float(ambient_intensity), 0.03))
        relighting_map = (total_shading + float(ambient_intensity)) * subject_mask + bg_backdrop * (1.0 - subject_mask)
        relighting_map = torch.clamp(relighting_map, 0.0, 1.0)
    elif "Automotive" in framing or ("Auto" in framing and "Automotive" in effective_framing):
        # Commercial Automotive Infinity Cove (1280 x 768)
        # Giant curved cyclorama cove with dual overhead stripbox light banks
        res_h = 768
        res_w = 1280
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res_h, device=device),
            torch.linspace(-1.0, 1.0, res_w, device=device),
            indexing="ij"
        )
        # Parabolic cyclorama wall & floor perspective
        floor_grad = torch.clamp((y - 0.15) / 0.85, 0.0, 1.0)
        cove_curve = torch.clamp(1.0 - torch.abs(y - 0.15) * 1.2, 0.0, 1.0)

        # Key & Fill directional washes
        lx = key_vec[0].to(device)
        flx = fill_vec[0].to(device)
        key_wash = torch.clamp((x * lx + 0.6) / 1.4, 0.0, 1.0) * key_params["intensity"]
        fill_wash = torch.clamp((x * flx + 0.5) / 1.5, 0.0, 1.0) * fill_params["intensity"] * 0.45

        # Dual overhead elongated stripbox reflection banks
        strip1 = torch.exp(-((x - 0.28) / 0.14)**4) * torch.exp(-((y + 0.50) / 0.12)**2) * key_params["intensity"] * 1.2
        strip2 = torch.exp(-((x + 0.32) / 0.18)**4) * torch.exp(-((y + 0.48) / 0.10)**2) * fill_params["intensity"] * 0.8

        c_k = torch.tensor(key_col, device=device).view(1, 1, 3)
        c_f = torch.tensor(fill_col, device=device).view(1, 1, 3)
        overhead_white = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 1, 3)

        cove = (key_wash.unsqueeze(-1) * c_k + fill_wash.unsqueeze(-1) * c_f) * (0.45 + cove_curve.unsqueeze(-1) * 0.55)
        cove = cove + (strip1 + strip2).unsqueeze(-1) * overhead_white * 0.85
        cove = cove * (1.0 - (x**2).unsqueeze(-1) * 0.15)
        cove = cove + float(ambient_intensity)
        relighting_map = torch.clamp(cove.unsqueeze(0), 0.0, 1.0)
    else:
        # Volumetric Studio Cyc environment (curved wall + floor room gradient, 1152 x 896)
        res_h = 1152
        res_w = 896
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res_h, device=device),
            torch.linspace(-1.0, 1.0, res_w, device=device),
            indexing="ij"
        )
        lx = key_vec[0].to(device)
        ly = key_vec[1].to(device)
        key_grad = torch.clamp((x * lx - y * ly * 0.4 + 0.6) / 1.4, 0.0, 1.0) * key_params["intensity"]

        flx = fill_vec[0].to(device)
        fill_grad = torch.clamp((x * flx + 0.5) / 1.5, 0.0, 1.0) * fill_params["intensity"] * 0.4

        c_k = torch.tensor(key_col, device=device).view(1, 1, 3)
        c_f = torch.tensor(fill_col, device=device).view(1, 1, 3)
        vignette = 1.0 - (x**2 + y**2) * 0.22

        cyc = (key_grad.unsqueeze(-1) * c_k + fill_grad.unsqueeze(-1) * c_f) * vignette.unsqueeze(-1)
        cyc = cyc + float(ambient_intensity)
        relighting_map = torch.clamp(cyc.unsqueeze(0), 0.0, 1.0)

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
    bottom_half = draw_studio_radar_diagram(key_params, fill_params, rim_params, width=probe_res, height=half_h)
    bottom_half = bottom_half.to(device=device)

    # Complete square probe preview card
    probe_preview = torch.cat([top_half, bottom_half], dim=1)  # [1, probe_res, probe_res, 3]

    if return_passes:
        diff_out = (total_diffuse * subject_mask).cpu() if "total_diffuse" in locals() else relighting_map.cpu()
        spec_out = (total_specular * subject_mask).cpu() if "total_specular" in locals() else torch.zeros_like(relighting_map).cpu()
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

