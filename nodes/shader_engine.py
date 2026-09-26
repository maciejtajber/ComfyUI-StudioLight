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


def compute_light_shading(
    normals: torch.Tensor,
    light_vec: torch.Tensor,
    light_color: Tuple[float, float, float],
    intensity: float,
    modifier_name: str,
    depth: Optional[torch.Tensor] = None,
    return_separate: bool = False,
) -> torch.Tensor:
    """Computes diffuse and Blinn-Phong specular contribution for a single light."""
    preset = MODIFIER_PRESETS.get(modifier_name, MODIFIER_PRESETS["Octabox (Soft)"])
    wrap = min(preset.get("wrap_factor", 0.0), 0.45)
    spec_power = preset["specular_power"]
    spec_strength = preset["specular_strength"]
    falloff_exp = preset["falloff_exponent"]

    device = normals.device
    dtype = normals.dtype
    l_vec = light_vec.to(device=device, dtype=dtype).view(1, 1, 1, 3)

    # Dot product N . L
    n_dot_l = torch.sum(normals * l_vec, dim=-1, keepdim=True)

    # Wrap lighting (Half-Lambert)
    if wrap > 0.0:
        diffuse = torch.clamp((n_dot_l + wrap) / (1.0 + wrap), min=0.0, max=1.0)
    else:
        diffuse = torch.clamp(n_dot_l, min=0.0, max=1.0)

    if falloff_exp != 1.0:
        diffuse = torch.pow(diffuse, falloff_exp)

    # View vector (towards camera +Z)
    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3)
    h_vec = l_vec + v_vec
    h_norm = torch.norm(h_vec, dim=-1, keepdim=True) + 1e-7
    h_vec = h_vec / h_norm

    # Blinn-Phong Specular (controlled to preserve diffuse contrast)
    n_dot_h = torch.clamp(torch.sum(normals * h_vec, dim=-1, keepdim=True), min=0.0, max=1.0)
    specular = torch.pow(n_dot_h, spec_power) * spec_strength * 0.35

    shading = (diffuse + specular) * float(intensity)
    c_tensor = torch.tensor(light_color, device=device, dtype=dtype).view(1, 1, 1, 3)
    diffuse_rgb = diffuse * float(intensity) * c_tensor
    specular_rgb = specular * float(intensity) * c_tensor

    if depth is not None:
        d = depth.to(device=device, dtype=dtype)
        if d.shape[-1] != 1 and len(d.shape) == 4:
            d = d[..., 0:1]
        attenuation = torch.clamp(1.0 - (d * 0.35), min=0.25, max=1.0)
        diffuse_rgb = diffuse_rgb * attenuation
        specular_rgb = specular_rgb * attenuation

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

    # Helper to draw light vector
    def draw_vector(azimuth: float, elevation: float, color_rgb: Tuple[int, int, int], label: str, enabled: bool):
        if not enabled:
            return
        # Azimuth: 0° is bottom (camera), +45° is camera-right (screen right), +/-180° is top (backlight)
        ang_rad = math.radians(azimuth)
        lx = cx + int(radius * math.sin(ang_rad))
        ly = cy + int(radius * math.cos(ang_rad))

        # Ray line
        draw.line([cx, cy, lx, ly], fill=color_rgb, width=3)
        # Light head node
        draw.ellipse([lx - 9, ly - 9, lx + 9, ly + 9], fill=color_rgb, outline=(255, 255, 255), width=2)
        # Degree label
        tx = lx + (10 if math.sin(ang_rad) >= 0 else -56)
        ty = ly + (6 if math.cos(ang_rad) >= 0 else -14)
        draw.text((tx, ty), f"{label} {int(azimuth)}°", fill=color_rgb)

    # Draw Key (Amber), Fill (Cyan), Rim (Magenta)
    draw_vector(key_params["azimuth"], key_params["elevation"], (255, 185, 45), "KEY", key_params.get("enabled", True))
    draw_vector(fill_params["azimuth"], fill_params["elevation"], (65, 205, 255), "FILL", fill_params.get("enabled", True))
    draw_vector(rim_params["azimuth"], rim_params["elevation"], (245, 75, 175), "RIM", rim_params.get("enabled", True))

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
    preview_resolution: int = 512,
    framing: str = "Portrait (Classical Bust)",
    detected_prompt: str = "",
    return_passes: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Renders the complete 3-point photographic studio lighting pass and dual-probe preview.
    Returns:
      relighting_map: [B, H, W, 3] in range [0, 1]
      probe_preview:  [1, H, W, 3] in range [0, 1] (Grey ball + Chrome ball + 2D Radar)
    """
    device = get_preferred_device() if normals is None else normals.device

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
        if invert_normals_y:
            norm_unpacked[..., 1] = -norm_unpacked[..., 1]

        norm_len = torch.norm(norm_unpacked, dim=-1, keepdim=True) + 1e-7
        norm_unit = norm_unpacked / norm_len

        # Precise solid alpha mask (eliminates holes 100%)
        if asset_mask is not None:
            subject_mask = asset_mask
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
                    norm_unit, p_vec, p_col, p_dict["intensity"], p_dict["modifier"], depth, return_separate=True
                )
                total_diffuse = total_diffuse + d_rgb
                total_specular = total_specular + s_rgb

        total_shading = total_diffuse + total_specular

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

    # Matte 18% Grey Ball
    grey_shd = torch.zeros_like(sphere_normals)
    if key_params.get("enabled", True):
        grey_shd = grey_shd + compute_light_shading(sphere_normals, key_vec, key_col, key_params["intensity"], key_params["modifier"])
    if fill_params.get("enabled", True):
        grey_shd = grey_shd + compute_light_shading(sphere_normals, fill_vec, fill_col, fill_params["intensity"], fill_params["modifier"])
    if rim_params.get("enabled", True):
        grey_shd = grey_shd + compute_light_shading(sphere_normals, rim_vec, rim_col, rim_params["intensity"], rim_params["modifier"])
    grey_ball = torch.clamp((grey_shd + ambient_intensity) * sphere_mask * 0.72, 0.0, 1.0)

    # Mirror Chrome Ball
    chrome_ball = torch.zeros_like(sphere_normals)
    v_vec = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 1, 3)
    n_dot_v = torch.sum(sphere_normals * v_vec, dim=-1, keepdim=True)
    refl_rays = 2.0 * n_dot_v * sphere_normals - v_vec
    refl_rays = refl_rays / (torch.norm(refl_rays, dim=-1, keepdim=True) + 1e-7)

    for p, vec, col in [
        (key_params, key_vec, key_col),
        (fill_params, fill_vec, fill_col),
        (rim_params, rim_vec, rim_col),
    ]:
        if p.get("enabled", True):
            lv = vec.to(device=device).view(1, 1, 1, 3)
            r_dot_l = torch.clamp(torch.sum(refl_rays * lv, dim=-1, keepdim=True), min=0.0, max=1.0)
            spec_spot = torch.pow(r_dot_l, 100.0) * float(p["intensity"]) * 2.2
            col_t = torch.tensor(col, device=device).view(1, 1, 1, 3)
            chrome_ball = chrome_ball + spec_spot * col_t

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
