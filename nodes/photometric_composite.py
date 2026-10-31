"""
ComfyUI-StudioLight — Adaptive Scene-Aware Photometric Composite Node
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Transfers studio lighting gains onto linearized sRGB draft images. Generated or
processed drafts are not calibrated scene radiance. Includes:
  1. Depth-Clamped 3-Channel RGB Guided Filter with Anti-Halo Boundary Guard
  2. Normalized Masked Intrinsic Crevice De-Lighting (Scale-Adaptive, Zero BG Bleed)
  3. Sub-Black Achromatic & Matte-Fabric Gate (Zero Coat Stains at < 0.50 Denoise)
  4. 3-Domain Physical Gain Equation with Studio Cyclorama Coupling & Halo-Free Sky Gate
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from .photometric_support import (
    image_tensor, resize_image, scalar, soft_light_blend, distant_background_weight, quantile,
    enclosure_bounce, recover_masked_shadows, contact_sheet,
)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Accurate IEC 61966-2-1 sRGB EOTF (Linear Rec.709)."""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow(torch.clamp((x + 0.055) / 1.055, min=1e-7), 2.4),
    )


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Accurate IEC 61966-2-1 sRGB OETF."""
    x_clamped = torch.clamp(x, 0.0, 1.0)
    return torch.where(
        x_clamped <= 0.0031308,
        x_clamped * 12.92,
        1.055 * torch.pow(torch.clamp(x_clamped, min=1e-7), 1.0 / 2.4) - 0.055,
    )


SPEC_PASS_REFERENCE = 0.25


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """Rec.709 / sRGB Photometric Luminance [B, H, W, 1]."""
    weights = torch.tensor([0.212639, 0.715169, 0.072192], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    return torch.sum(x * weights, dim=-1, keepdim=True)


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """Hermite cubic smoothstep in [0, 1]."""
    t = torch.clamp((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def filmic_reinhard_tonemap(x: torch.Tensor, white_point: float = 3.5) -> torch.Tensor:
    """
    Photographic rational shoulder tone reproduction.
    Preserves 100% 1:1 linear contrast and saturation on standard values (<= 0.90),
    gently rolling off specular overshoots and highlights into display headroom.
    """
    threshold = 0.90
    headroom = 1.0 - threshold
    excess = torch.clamp(x - threshold, min=0.0)

    we = max(float(white_point) - threshold, 0.2)
    term = excess / we
    compressed_excess = headroom * (1.0 - 1.0 / (1.0 + (excess / headroom) * (1.0 + term)))
    return torch.where(x > threshold, threshold + compressed_excess, x)


def luminance_hdr_tonemap(x: torch.Tensor, white_point: float = 3.5) -> torch.Tensor:
    """Map luminance, then compress chroma toward grey to fit the display gamut."""
    x = x.clamp_min(0)
    lum = rgb_to_luminance(x)
    toned_lum = filmic_reinhard_tonemap(lum, white_point=white_point)
    # Preserve sub-black values too (no epsilon-induced darkening).
    mapped = x * torch.where(lum > 0, toned_lum / lum.clamp_min(torch.finfo(x.dtype).tiny), torch.ones_like(lum))
    delta = mapped - toned_lum
    eps = torch.finfo(x.dtype).tiny
    upper = torch.where(delta > 0, (1 - toned_lum) / delta.clamp_min(eps), torch.ones_like(delta))
    lower = torch.where(delta < 0, toned_lum / (-delta).clamp_min(eps), torch.ones_like(delta))
    saturation = torch.minimum(upper, lower).amin(dim=-1, keepdim=True).clamp(0, 1)
    return (toned_lum + saturation * delta).clamp(0, 1)


def _box_filter_bchw(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Separable spatial box average with replicate padding on BCHW tensors."""
    if radius <= 0:
        return x
    k = 2 * radius + 1
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(F.avg_pool2d(x_pad, kernel_size=(1, k), stride=1), kernel_size=(k, 1), stride=1)


def depth_clamped_rgb_guided_filter(
    mask_bhw1: torch.Tensor,
    guide_rgb_bhw3: torch.Tensor,
    d_norm_bhw1: torch.Tensor,
    d_split: float,
    radius: int = 5,
    eps: float = 1e-3,
) -> torch.Tensor:
    """
    O(N) 3-Channel RGB Joint Guided Filter with:
      1. 7px Depth-Gradient Guardband & hard_bg Lock (prevents sawtooth mortar bleed).
      2. Anti-Bright-Halo Silhouette Pullback (prevents foreground strobe/rim gain from
         creating a glowing white halo on bright walls/cycloramas around dark hair or coats).
    """
    p = mask_bhw1.permute(0, 3, 1, 2).contiguous()       # [B, 1, H, W]
    I = guide_rgb_bhw3.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]

    H_g, W_g = guide_rgb_bhw3.shape[1], guide_rgb_bhw3.shape[2]
    eff_r = radius if min(H_g, W_g) >= 64 else max(1, min(radius, min(H_g, W_g) // 16))

    mean_I = _box_filter_bchw(I, eff_r)
    mean_p = _box_filter_bchw(p, eff_r)
    mean_Ip = _box_filter_bchw(I * p, eff_r)
    cov_Ip = mean_Ip - mean_I * mean_p

    var_I = _box_filter_bchw(I * I, eff_r) - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - (a * mean_I).mean(dim=1, keepdim=True)

    mean_a = _box_filter_bchw(a, eff_r) / 3.0
    mean_b = _box_filter_bchw(b, eff_r)
    q = (mean_a * I).sum(dim=1, keepdim=True) + mean_b
    q_bhw1 = torch.clamp(q.permute(0, 2, 3, 1).contiguous(), 0.0, 1.0)

    # Local depth-discontinuity band around silhouette
    d_bchw = d_norm_bhw1.permute(0, 3, 1, 2).contiguous()
    pad_d = 3 if min(H_g, W_g) >= 64 else 1
    k_d = 2 * pad_d + 1
    d_max = F.max_pool2d(F.pad(d_bchw, (pad_d, pad_d, pad_d, pad_d), mode="replicate"), kernel_size=k_d, stride=1)
    d_min = -F.max_pool2d(F.pad(-d_bchw, (pad_d, pad_d, pad_d, pad_d), mode="replicate"), kernel_size=k_d, stride=1)
    d_grad_band = torch.clamp(((d_max - d_min) - 0.04) / 0.06, 0.0, 1.0).permute(0, 2, 3, 1).contiguous()

    soft_bg = 1.0 - _smoothstep(d_split - 0.069, d_split - 0.02, d_norm_bhw1)
    soft_fg = _smoothstep(d_split + 0.06, d_split + 0.18, d_norm_bhw1) * _smoothstep(0.70, 0.90, mask_bhw1)

    snapped = d_grad_band * q_bhw1 + (1.0 - d_grad_band) * mask_bhw1
    snapped = snapped * (1.0 - soft_bg)
    snapped = torch.maximum(snapped, soft_fg)

    # Anti-Bright-Halo Pullback:
    # At silhouette boundaries where a dark subject (hair/suit) meets a brighter background wall,
    # pull intermediate mask values (0 < m_fg < 0.75) inward so high foreground strobe gains (>1.0)
    # never bleed onto the bright background wall as a glowing white outline.
    guide_lum = rgb_to_luminance(guide_rgb_bhw3).permute(0, 3, 1, 2).contiguous()
    fg_lum_num = _box_filter_bchw(guide_lum * (p > 0.6).float(), 5)
    fg_lum_den = torch.clamp(_box_filter_bchw((p > 0.6).float(), 5), min=1e-4)
    local_fg_lum = (fg_lum_num / fg_lum_den).permute(0, 2, 3, 1).contiguous()
    pixel_lum = guide_lum.permute(0, 2, 3, 1).contiguous()

    is_brighter_bg_edge = _smoothstep(0.04, 0.18, pixel_lum - local_fg_lum) * d_grad_band
    snapped_inward = torch.clamp((snapped - 0.25) / 0.75, 0.0, 1.0)
    snapped = (1.0 - is_brighter_bg_edge) * snapped + is_brighter_bg_edge * (snapped_inward * snapped_inward)

    return torch.clamp(snapped, 0.0, 1.0)


def sanitize_subblack_and_delight(
    base_linear: torch.Tensor,
    m_fg: torch.Tensor,
    r_local: int = 14,
    r_broad: int = 42,
    beta: float = 0.30,
    black_gate: str = "Legacy material protection",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    1. Sub-Black Achromatic Gate:
       Strips warm Pass 1 VAE quantization noise in deep shadows (L < 0.032) so black coats
       never develop brown stains at < 0.50 denoise.
    2. In-Scene Practical Emitter Gate (`m_emissive`):
       Detects self-luminous props (caveman torch flames, smartphone screens, neon tubes)
       inside `m_fg` so they are never snuffed out by studio shadow attenuation.
    3. Bidirectional Masked De-Lighting + Practical Glare Neutralizer:
       Both lifts Pass 1 baked dark crevices (l_broad > l_local) AND gently compresses
       baked practical glare hotspots (l_local > l_broad, e.g. smartphone/torch glare on skin)
       while neutralizing localized color burns toward the subject's broad chromaticity.
    Returns:
       (base_delit, dark_material_gate, m_emissive)
    """
    if base_linear.shape[0] > 1:
        frames = [sanitize_subblack_and_delight(base_linear[i:i+1], m_fg[i:i+1],
                  r_local, r_broad, beta, black_gate) for i in range(base_linear.shape[0])]
        return tuple(torch.cat([frame[j] for frame in frames], dim=0) for j in range(3))
    lum = rgb_to_luminance(base_linear)  # [B, H, W, 1]

    # Gate 1: Achromatize sub-black foreground pixels (L < 0.032)
    low_signal = black_gate == "Low-signal chroma preservation"
    chroma_keep = _smoothstep(0.0010, 0.0045, lum) if low_signal else _smoothstep(0.006, 0.032, lum)
    fg_achroma_mix = 1.0 - m_fg * (1.0 - chroma_keep)
    base_clean = fg_achroma_mix * base_linear + (1.0 - fg_achroma_mix) * lum

    # Gate 2: Dark-Material Protection Gate (strictly pointwise, no spatial box-window halos)
    cmax = base_clean.amax(dim=-1, keepdim=True)
    cmin = base_clean.amin(dim=-1, keepdim=True)
    pix_sat = (cmax - cmin) / (cmax + 1e-4)

    if low_signal:
        dark_material_gate = _smoothstep(0.0018, 0.0075, lum)
    else:
        is_dark_achromatic_garment = (
            (1.0 - _smoothstep(0.015, 0.060, lum))
            * (1.0 - _smoothstep(0.20, 0.40, pix_sat))
        )
        dark_material_gate = _smoothstep(0.003, 0.014, lum) * (1.0 - 0.90 * is_dark_achromatic_garment)

    # Gate 3: Practical Emitter Detector (torch, phone screen, bright neon)
    m_emissive = _smoothstep(0.55, 0.85, lum)

    if float(beta) <= 0.0:
        return base_clean, dark_material_gate, m_emissive

    # Strictly POINTWISE crevice delighting (ZERO spatial box/pool filters = ZERO square blocks)
    toe = torch.clamp((0.20 - lum) / 0.20, 0.0, 1.0)
    smooth_toe = toe * toe * (3.0 - 2.0 * toe)
    delight_weight = m_fg * dark_material_gate * (1.0 - m_emissive)
    gain = 1.0 + float(beta) * delight_weight * smooth_toe
    base_delit = torch.clamp(base_clean * gain, 0.0, 1.0)

    return base_delit, dark_material_gate, m_emissive


def intrinsic_crevice_delight(
    base_linear: torch.Tensor,
    m_fg: torch.Tensor,
    r_local: int = 14,
    r_broad: int = 42,
    beta: float = 0.30,
) -> torch.Tensor:
    """Backward-compatible alias for sanitize_subblack_and_delight returning base_delit."""
    base_delit, _, _ = sanitize_subblack_and_delight(base_linear, m_fg, r_local=r_local, r_broad=r_broad, beta=beta)
    return base_delit


def measure_photometric_environment(
    base_linear: torch.Tensor,
    d_norm: torch.Tensor,
    d_split: float,
    m_fg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Upgraded Photometric Light Meter:
      - l_env: [B, 1, 1, 1] 60th percentile background luminance.
      - c_env_field: [B, H, W, 3] spatially-varying ambient chromaticity field (supports multi-color Disco/Neon wrap!).
      - m_sky: [B, H, W, 1] 1-pixel accurate Sky Isolation Gate.
      - m_fog_haze: [B, H, W, 1] Non-Solid Participating Media Gate (Fog, Smoke, Disco Haze).
      - is_studio_cyc: [B, 1, 1, 1] Depth-planar & smooth studio cyclorama detector.
    """
    lum = rgb_to_luminance(base_linear)  # [B, H, W, 1]
    bg_mask_bhw1 = ((d_norm < d_split) & (m_fg < 0.20)).float()
    bg_mask = bg_mask_bhw1.squeeze(-1) > 0.5

    # 1. Halo-Free Per-Pixel Sky Isolation Gate M_sky(x)
    max_c = base_linear.max(dim=-1, keepdim=True).values
    min_c = base_linear.min(dim=-1, keepdim=True).values
    sat = (max_c - min_c) / torch.clamp(max_c, min=1e-4)

    far_depth_gate = _smoothstep(0.03, 0.09, 0.11 - d_norm)
    bright_sky_gate = _smoothstep(0.22, 0.45, lum)
    low_sat_gate = _smoothstep(0.0, 0.45, 0.65 - sat)
    m_sky = torch.clamp(far_depth_gate * bright_sky_gate * low_sat_gate, 0.0, 1.0)

    # 2. High-frequency micro-texture energy (solid walls/stone have texture; fog/smoke/cyc are smooth)
    lum_bchw = lum.permute(0, 3, 1, 2).contiguous()
    lum_blur = _box_filter_bchw(lum_bchw, 2).permute(0, 2, 3, 1).contiguous()
    hf_energy = torch.abs(lum - lum_blur)
    local_hf_mean = _box_filter_bchw(hf_energy.permute(0, 3, 1, 2).contiguous(), 12).permute(0, 2, 3, 1).contiguous()

    # 3. Spatially-Varying Ambient Chromaticity Field `c_env_field` [B, H, W, 3]
    # Preserves left-vs-right neon/disco color zones with smooth continuous fallback into subject
    bg_w_bchw = bg_mask_bhw1.permute(0, 3, 1, 2).contiguous()
    rgb_bchw = base_linear.permute(0, 3, 1, 2).contiguous()
    bg_coverage = _box_filter_bchw(bg_w_bchw, 64)
    broad_bg_rgb = _box_filter_bchw(rgb_bchw * bg_w_bchw, 64) / torch.clamp(bg_coverage, min=1e-4)
    broad_bg_lum = _box_filter_bchw(lum_bchw * bg_w_bchw, 64) / torch.clamp(bg_coverage, min=1e-4)
    local_c_env = (broad_bg_rgb / torch.clamp(broad_bg_lum, min=1e-3)).permute(0, 2, 3, 1).contiguous().clamp(0.50, 2.00)

    # Global background chromaticity fallback for the interior of foreground subjects
    valid_bg_count = bg_w_bchw.sum(dim=(2, 3), keepdim=True)
    global_bg_rgb = (rgb_bchw * bg_w_bchw).sum(dim=(2, 3), keepdim=True) / torch.clamp(valid_bg_count, min=1.0)
    global_bg_lum = (lum_bchw * bg_w_bchw).sum(dim=(2, 3), keepdim=True) / torch.clamp(valid_bg_count, min=1.0)
    global_c_env = (global_bg_rgb / torch.clamp(global_bg_lum, min=1e-3)).permute(0, 2, 3, 1).contiguous().clamp(0.50, 2.00)

    # Smooth Hermite transition between localized edge chromaticity and global ambient fallback
    w_local = _smoothstep(0.005, 0.15, bg_coverage.permute(0, 2, 3, 1).contiguous())
    c_env_field = torch.clamp(w_local * local_c_env + (1.0 - w_local) * global_c_env, min=0.50, max=2.00)

    B, H, W, _ = lum.shape
    l_env_list = []
    cyc_list = []

    for b_idx in range(B):
        valid_bg = bg_mask[b_idx]
        bg_lum_vals = lum[b_idx, :, :, 0][valid_bg]
        if bg_lum_vals.numel() > 32:
            step = max(1, bg_lum_vals.numel() // 16384)
            l_env_b = quantile(bg_lum_vals[::step], 0.60)

            non_sky_bg = valid_bg & (m_sky[b_idx, :, :, 0] < 0.25)
            if non_sky_bg.sum() > 64:
                hf_med = quantile(hf_energy[b_idx, :, :, 0][non_sky_bg][::step], 0.75)
                sat_med = quantile(sat[b_idx, :, :, 0][non_sky_bg][::step], 0.75)
                sky_frac = m_sky[b_idx, :, :, 0][valid_bg].mean()
                # Depth IQR planarity gate ensures deep foggy scenes or corridors aren't mistaken for studio paper
                bg_depths = d_norm[b_idx, :, :, 0][non_sky_bg][::step]
                d_iqr = quantile(bg_depths, 0.75) - quantile(bg_depths, 0.25)

                bg_depth_med = quantile(bg_depths, 0.50)
                smooth_score = torch.clamp((0.018 - hf_med) / 0.012, 0.0, 1.0)
                neutral_score = torch.clamp((0.22 - sat_med) / 0.14, 0.0, 1.0)
                no_sky_score = torch.clamp((0.08 - sky_frac) / 0.06, 0.0, 1.0)
                planar_score = torch.clamp((0.10 - d_iqr) / 0.06, 0.0, 1.0)
                close_wall_score = torch.clamp(1.0 - (d_split - bg_depth_med - 0.12) / 0.12, 0.0, 1.0)
                lit_cyc_score = torch.clamp((l_env_b - 0.025) / 0.020, 0.0, 1.0)
                cyc_b = smooth_score * neutral_score * no_sky_score * planar_score * close_wall_score * lit_cyc_score
            else:
                cyc_b = torch.tensor(0.0, device=base_linear.device, dtype=base_linear.dtype)
        else:
            l_env_b = lum[b_idx].mean()
            cyc_b = torch.tensor(0.0, device=base_linear.device, dtype=base_linear.dtype)

        l_env_list.append(l_env_b)
        cyc_list.append(cyc_b)

    l_env = torch.stack(l_env_list, dim=0).view(B, 1, 1, 1)
    is_studio_cyc = torch.stack(cyc_list, dim=0).view(B, 1, 1, 1)

    # 4. Non-Solid Participating Media Gate (`m_fog_haze`) for Fog, Smoke, and Disco Haze:
    # Triggers where background is ultra-smooth (low micro-texture) and far/mid-distant,
    # and NOT a neutral planar studio cyclorama. Works on both bright fog and colored club smoke!
    smooth_air_gate = _smoothstep(0.004, 0.016, 0.020 - local_hf_mean)
    depth_air_gate = _smoothstep(0.02, 0.18, (d_split - 0.04) - d_norm)
    m_fog_haze = torch.clamp(smooth_air_gate * depth_air_gate * (1.0 - is_studio_cyc) * (1.0 - m_fg), 0.0, 1.0)

    return l_env, c_env_field, m_sky, m_fog_haze, is_studio_cyc


def enclosed_wall_irradiance(
    depth: torch.Tensor,
    rays_z1: torch.Tensor,
    normals: torch.Tensor,
    light_position: torch.Tensor,
    flux_rgb: torch.Tensor,
    wall_mask: torch.Tensor,
    sky_mask: torch.Tensor,
    visibility: torch.Tensor,
    geometry_confidence: torch.Tensor,
    *,
    z_near: float = 0.8,
    z_far: float = 3.5,
    beam_pdf: float = 1.0 / (4.0 * math.pi),
    r_min: float = 0.05,
) -> torch.Tensor:
    """
    Direct irradiance on enclosed room walls from finite harmonic depth mapping:
      z(d) = (d / z_near + (1 - d) / z_far)^-1
      E_direct = M_wall * (1 - M_sky) * c_geom * V * (Phi_rgb * beam_pdf / max(r^2, r_min^2)) * max(N . l, 0)
    RGB tensors: [..., 3]; depth/masks/visibility: [..., 1].
    Camera convention: positive Z forward; rays_z1[..., 2] == 1.
    Positions and normals must share that coordinate system.
    flux_rgb: linear RGB radiant-power proxy.
    beam_pdf: normalized emission density toward each receiving point.
    """
    if not (0 < z_near <= z_far and r_min > 0):
        raise ValueError("Require 0 < z_near <= z_far and r_min > 0")
    d = depth.clamp(0, 1)
    z = 1.0 / (d / z_near + (1.0 - d) / z_far)
    position = rays_z1 * z
    to_light = light_position - position
    distance2 = to_light.square().sum(-1, keepdim=True)
    direction = F.normalize(to_light, dim=-1, eps=1e-8)
    normal = F.normalize(normals, dim=-1, eps=1e-8)
    cosine = (normal * direction).sum(-1, keepdim=True).clamp_min(0)
    support = (
        wall_mask.clamp(0, 1)
        * (1.0 - sky_mask.clamp(0, 1))
        * geometry_confidence.clamp(0, 1)
        * visibility.clamp(0, 1)
    )
    intensity_rgb = flux_rgb * beam_pdf
    return support * cosine * intensity_rgb / distance2.clamp_min(r_min**2)


def silhouette_strobe_transport(
    base_linear: torch.Tensor,
    ambient_gain_rgb: torch.Tensor,
    albedo_prior: torch.Tensor,
    illumination_confidence: torch.Tensor,
    subject_mask: torch.Tensor,
    flash_irradiance_rgb: torch.Tensor,
    *,
    exposure_gain: float = 1.0,
    noise_floor: float = 1e-4,
) -> torch.Tensor:
    """
    Intrinsic albedo decomposition & reflectance-proportional strobe radiance transport:
      rho_hat = c * clip(B / (S_A + eps), 0, 1) + (1 - c) * rho_prior
      L_proxy = B + M_subject * k * rho_hat * E_flash / pi
    Prevents the multiplicative darkening inversion trap on backlit silhouettes
    without flattening distinct materials into a bald mannequin mask.
    """
    if noise_floor <= 0 or exposure_gain < 0:
        raise ValueError("Require positive noise_floor and nonnegative exposure")
    weights = base_linear.new_tensor([0.212639, 0.715169, 0.072192]).view(1, 1, 1, 3)
    luminance = (base_linear * weights).sum(-1, keepdim=True)
    t = ((luminance - noise_floor) / (3.0 * noise_floor)).clamp(0, 1)
    signal_confidence = t * t * (3.0 - 2.0 * t)
    illumination_valid = (
        ambient_gain_rgb.amin(-1, keepdim=True) > noise_floor
    ).to(base_linear.dtype)
    confidence = (
        illumination_confidence.clamp(0, 1)
        * signal_confidence
        * illumination_valid
    )
    observed_albedo = (
        base_linear / ambient_gain_rgb.clamp_min(noise_floor)
    ).clamp(0, 1)
    albedo = (
        confidence * observed_albedo
        + (1.0 - confidence) * albedo_prior.clamp(0, 1)
    )
    flash_radiance = (
        exposure_gain * albedo * flash_irradiance_rgb.clamp_min(0) / math.pi
    )
    return base_linear + subject_mask.clamp(0, 1) * flash_radiance


class StudioPhotometricComposite:
    """
    Composites 3D deferred relighting maps onto base/draft images in physical linear space
    using the 3-Domain Environmental Lighting Equation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lighting_map": ("IMAGE",),
                "lighting_strength": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.0,
                    "max": 2.5,
                    "step": 0.05,
                }),
                "ambient_floor": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.00,
                    "step": 0.01,
                    "tooltip": "Scene-referred Linear Rec.709 floor (default 0.00 so StudioLightRig.ambient_intensity controls true shadow density without double-flooring).",
                }),
                "specular_kick": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                }),
                "chroma_bleed": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "filmic_white_point": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.25,
                }),
            },
            "optional": {
                "specular_map": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "wall_shadow_density": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 0.95,
                    "step": 0.05,
                    "tooltip": "3D cast shadow density on background walls (0.65 retains 35% natural ambient bounce).",
                }),
                "bg_exposure_ev": ("FLOAT", {
                    "default": 0.0,
                    "min": -3.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Background ambient exposure offset in EV stops (0.0 preserves Pass 1 native exposure 1:1).",
                }),
                "crevice_delight": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.0,
                    "max": 0.65,
                    "step": 0.05,
                    "tooltip": "Lifts Pass 1 baked facial crevices (scale-adaptive, masked to FG) before applying 3D studio AO.",
                }),
                "atmospheric_haze": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Strobe scatter in fog, smoke, or disco haze. Zero disables scatter; the image heuristic modulates the selected amount.",
                }),
                "telemetry": ("STUDIOLIGHT_TELEMETRY",),
                "room_bounce_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Optional additive visible-wall fill heuristic. Requires current rig telemetry; not metric global illumination."}),
                "wall_reflectance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enclosure_override": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "-1: relative-depth estimate; 0: open; 1: enclosed. Still requires visible wall samples."}),
                "recovery_mask": ("MASK",),
                "emission_mask": ("MASK", {"tooltip": "Optional explicit light-source mask: white preserves source emission. Unconnected disables emission preservation. Skin and reflective highlights are not light sources."}),
                "shadow_recovery": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Optional bounded reconstruction from exposed pixels of ONE material selected by recovery_mask. No automatic skin colour."}),
                "black_gate": (["Legacy material protection", "Low-signal chroma preservation"],),
                "tone_mapping": (["Legacy per-channel", "Luminance HDR"],),
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.1}),
                "background_protection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lighting_map_includes_specular": ("BOOLEAN", {"default": True}),
                "compositing_mode": (["Linear HDR", "Soft Light"],),
                "blend_amount": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("lit_image", "diagnostic_contact_sheet")
    FUNCTION = "execute_composite"
    CATEGORY = "StudioLight/Photometric"
    DESCRIPTION = (
        "Composites 3D deferred physical lighting maps onto base/draft images in linear Rec.709 HDR space.\n"
        "Features 3-Domain scene-aware environmental preservation, Depth-Clamped RGB Guided Filtering,\n"
        "Normalized Masked Crevice De-Lighting, Sub-Black Coat Protection, and Filmic Reinhard tone compression."
    )

    def execute_composite(
        self, *args, auto_exposure_lift_dark_scenes=False, erase_wall_cast_shadows=False, **kwargs
    ):
        return self.composite(
            *args,
            auto_exposure_lift_dark_scenes=auto_exposure_lift_dark_scenes,
            erase_wall_cast_shadows=erase_wall_cast_shadows,
            **kwargs,
        )

    def composite(
        self, image, lighting_map, lighting_strength=1.2, ambient_floor=0.0,
        specular_kick=0.45, chroma_bleed=0.85, filmic_white_point=3.5,
        specular_map=None, depth_map=None, wall_shadow_density=0.65,
        bg_exposure_ev=0.0, crevice_delight=0.30, atmospheric_haze=0.0,
        telemetry=None, room_bounce_strength=0.0, wall_reflectance=0.5,
        enclosure_override=-1.0, normals=None,
        auto_exposure_lift_dark_scenes=False, erase_wall_cast_shadows=False,
        **kwargs
    ):
        """Validate once; process frames independently, preserving per-frame telemetry.

        Singleton batches expand explicitly. Other unequal batch sizes are errors.
        New typed telemetry wins over the legacy tensor attribute.
        """
        recovery_mask = kwargs.get("recovery_mask", None)
        shadow_recovery = kwargs.get("shadow_recovery", 0.0)
        emission_mask = kwargs.get("emission_mask", None)
        black_gate = kwargs.get("black_gate", "Legacy material protection")
        tone_mapping = kwargs.get("tone_mapping", "Legacy per-channel")
        exposure_ev = kwargs.get("exposure_ev", 0.0)
        background_protection = kwargs.get("background_protection", 0.0)
        lighting_map_includes_specular = kwargs.get("lighting_map_includes_specular", True)
        compositing_mode = kwargs.get("compositing_mode", "Linear HDR")
        blend_amount = kwargs.get("blend_amount", 0.35)
        wall_transport = kwargs.get("wall_transport", None)
        return_repair_mask = kwargs.get("return_repair_mask", False)
        options = dict(lighting_strength=lighting_strength, ambient_floor=ambient_floor,
                       specular_kick=specular_kick, chroma_bleed=chroma_bleed,
                       filmic_white_point=filmic_white_point, wall_shadow_density=wall_shadow_density,
                       bg_exposure_ev=bg_exposure_ev, crevice_delight=crevice_delight,
                       atmospheric_haze=atmospheric_haze, room_bounce_strength=room_bounce_strength,
                       wall_reflectance=wall_reflectance, enclosure_override=enclosure_override,
                       shadow_recovery=shadow_recovery, exposure_ev=exposure_ev,
                       background_protection=background_protection, blend_amount=blend_amount)
        bounds = dict(lighting_strength=(0, 2.5), ambient_floor=(0, 1), specular_kick=(0, 3),
                      chroma_bleed=(0, 1), filmic_white_point=(1, 10), wall_shadow_density=(0, .95),
                      bg_exposure_ev=(-3, 2), crevice_delight=(0, .65), atmospheric_haze=(0, 1),
                      room_bounce_strength=(0, 1), wall_reflectance=(0, 1), enclosure_override=(-1, 1),
                      shadow_recovery=(0, 1), exposure_ev=(-8, 8), background_protection=(0, 1), blend_amount=(0, 1))
        for name, value in options.items():
            options[name] = scalar(value, name, *bounds[name])
        for name, value, choices in (
            ("tone_mapping", tone_mapping, ("Legacy per-channel", "Luminance HDR")),
            ("compositing mode", compositing_mode, ("Linear HDR", "Soft Light")),
            ("black_gate", black_gate, ("Legacy material protection", "Low-signal chroma preservation")),
        ):
            if value not in choices:
                raise ValueError(f"Unknown {name}: {value}")
        if background_protection > 0 and depth_map is None:
            raise ValueError("background_protection requires a depth_map")
        if shadow_recovery > 0 and recovery_mask is None:
            raise ValueError("shadow_recovery requires a recovery_mask selecting one material")
        options.update(black_gate=black_gate, tone_mapping=tone_mapping,
                       compositing_mode=compositing_mode,
                       lighting_map_includes_specular=bool(lighting_map_includes_specular),
                       wall_transport=wall_transport,
                       auto_exposure_lift_dark_scenes=bool(auto_exposure_lift_dark_scenes),
                       erase_wall_cast_shadows=bool(erase_wall_cast_shadows))
        if not isinstance(image, torch.Tensor):
            raise ValueError("image must be a BHWC tensor")
        device = image.device
        if telemetry is None:
            telemetry = getattr(lighting_map, "_studiolight_telemetry", {})
        if not isinstance(telemetry, dict):
            raise ValueError("telemetry must be a StudioLight telemetry dictionary")
        if telemetry.get("depth_convention", "relative_near_one") != "relative_near_one":
            raise ValueError("telemetry depth_convention must be relative_near_one; metric depth requires calibration upstream")
        images = {"image": image_tensor(image, "image", device),
                  "lighting_map": image_tensor(lighting_map, "lighting_map", device)}
        for name, value in (("specular_map", specular_map), ("depth_map", depth_map), ("normals", normals)):
            if value is not None:
                images[name] = image_tensor(value, name, device, (1, 3) if name == "depth_map" else (3,))
        for name, mask in (("recovery_mask", recovery_mask), ("emission_mask", emission_mask)):
            if mask is not None:
                if isinstance(mask, torch.Tensor) and mask.ndim == 3:
                    mask = mask.unsqueeze(-1)
                images[name] = image_tensor(mask, name, device, (1,)).clamp(0, 1)
        maps = {}
        for name, channels in (("d_norm", (1,)), ("fg_confidence", (1,)), ("shadow_occlusion", (1,)),
                               ("volumetric_beam", (3,)), ("direct_diffuse", (3,)),
                               ("direct_specular", (3,)), ("ambient_rgb", (3,))):
            if telemetry.get(name) is not None:
                maps[name] = image_tensor(telemetry[name], "telemetry." + name, device, channels)
        sizes = [v.shape[0] for v in (*images.values(), *maps.values())]
        splits = torch.as_tensor(telemetry.get("d_split", .35), device=device, dtype=torch.float32).reshape(-1)
        if splits.numel() < 1 or not bool(torch.isfinite(splits).all()) or bool(((splits < 0) | (splits > 1)).any()):
            raise ValueError("telemetry.d_split must contain finite per-image thresholds in [0, 1]")
        sizes.append(splits.numel())
        batch = max(sizes)
        if any(n not in (1, batch) for n in sizes):
            raise ValueError(f"StudioLight: input batches must match or be 1; got {sizes}")
        size = images['image'].shape[1:3]
        images = {k: resize_image(v, size) for k, v in images.items()}
        maps = {k: resize_image(v, size).clamp_min(0) for k, v in maps.items()}
        for k in ("d_norm", "fg_confidence", "shadow_occlusion"):
            if k in maps:
                maps[k] = maps[k].clamp(0, 1)
        if bool(((images['image'] < 0) | (images['image'] > 1)).any()):
            raise ValueError("image must be display sRGB in [0, 1]")
        for k in ('lighting_map', 'specular_map'):
            if k in images:
                images[k] = images[k].clamp_min(0)
        results = []
        for i in range(batch):
            frame = {k: v[i:i+1] if v.shape[0] > 1 else v for k, v in images.items()}
            tele = {k: v[i:i+1] if v.shape[0] > 1 else v for k, v in maps.items()}
            tele['d_split'] = float(splits[i if splits.numel() > 1 else 0])
            # Automatically consume the matching raw pass from explicit telemetry.
            if 'specular_map' not in frame and 'direct_specular' in tele:
                frame['specular_map'] = tele['direct_specular']
            results.append(self._composite_single(**frame, telemetry=tele, **options))
        if return_repair_mask:
            return (
                torch.cat([r[0] for r in results], dim=0),
                torch.cat([r[1] for r in results], dim=0),
                torch.cat([r[2] for r in results], dim=0),
            )
        return tuple(torch.cat([r[j] for r in results], dim=0) for j in range(2))

    def _composite_single(
        self,
        image: torch.Tensor,
        lighting_map: torch.Tensor,
        lighting_strength: float = 1.2,
        ambient_floor: float = 0.0,
        specular_kick: float = 0.45,
        chroma_bleed: float = 0.85,
        filmic_white_point: float = 3.5,
        specular_map: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        wall_shadow_density: float = 0.65,
        bg_exposure_ev: float = 0.0,
        crevice_delight: float = 0.30,
        atmospheric_haze: float = 0.0,
        telemetry=None, room_bounce_strength=0.0, wall_reflectance=0.5,
        enclosure_override=-1.0, recovery_mask=None, shadow_recovery=0.0, emission_mask=None,
        black_gate="Legacy material protection", tone_mapping="Legacy per-channel",
        exposure_ev=0.0, background_protection=0.0, lighting_map_includes_specular=True,
        compositing_mode="Linear HDR", blend_amount=0.35,
        wall_transport=None, normals=None,
        auto_exposure_lift_dark_scenes=True, erase_wall_cast_shadows=True,
        **kwargs
    ):
        device = image.device
        img = image.to(device=device, dtype=torch.float32)
        light = lighting_map.to(device=device, dtype=torch.float32)
        telemetry = telemetry or {}

        H, W = img.shape[1], img.shape[2]
        appearance_repair_mask = torch.zeros_like(img[..., :1])

        # 1. Match spatial dimensions if needed
        if light.shape[1:3] != (H, W):
            light_resized = F.interpolate(
                light.permute(0, 3, 1, 2).contiguous(),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
        else:
            light_resized = light

        # 2. Linearize Base plate (IEC 61966-2-1 sRGB -> Linear Rec.709)
        base_linear = srgb_to_linear(img)

        # Stage 2B: Depth-Guided Exposure and Shadow-Lift Normalizer
        if auto_exposure_lift_dark_scenes and float(lighting_strength) > 0.0:
            initial_lum = rgb_to_luminance(base_linear)
            sample_lum = float(initial_lum.mean().item())
            if sample_lum < 0.28:
                target_lum = 0.28
                lift_ratio = min(2.8, max(1.0, math.sqrt(target_lum / max(sample_lum, 0.03))))
                shadow_t = torch.clamp((0.35 - initial_lum) / 0.35, 0.0, 1.0)
                is_high_peak = _smoothstep(0.40, 0.80, initial_lum)
                lift_curve = (lift_ratio - 1.0) * shadow_t * (1.0 - is_high_peak)
                base_linear = base_linear * (1.0 + lift_curve)

        if emission_mask is None:
            # Brightness/local contrast cannot distinguish a light source from
            # reflected light on skin. Preserve source emission only when the
            # caller supplies an explicit mask; never infer it from highlights.
            emission_mask = torch.zeros_like(base_linear[..., :1])
        # A current rig supplies separated components. Legacy combined maps can
        # subtract their matching raw specular pass before diffuse multiplication.
        strobe = telemetry.get("direct_diffuse")
        ambient_rgb = telemetry.get("ambient_rgb")
        if strobe is not None and ambient_rgb is not None:
            diffuse_input = strobe
            ambient_input = ambient_rgb
        else:
            diffuse_input = (light_resized - specular_map).clamp_min(0) if specular_map is not None and lighting_map_includes_specular else light_resized
            ambient_input = torch.zeros_like(diffuse_input)
        light_linear = diffuse_input * float(lighting_strength) + ambient_input
        zero = torch.zeros_like(img[..., :1])
        bounce_rgb = torch.zeros_like(img)
        bounce_confidence = zero
        shadow_occ_display = zero
        base_linear = recover_masked_shadows(base_linear, recovery_mask, shadow_recovery, rgb_to_luminance)

        # 3. Extract Telemetry from StudioLightRig (or optional depth_map fallback)
        d_norm = None
        d_split = 0.35
        fg_confidence = None
        shadow_occ = None

        if isinstance(telemetry, dict):
            d_norm = telemetry.get("d_norm")
            d_split = float(telemetry.get("d_split", 0.35))
            fg_confidence = telemetry.get("fg_confidence")
            shadow_occ = telemetry.get("shadow_occlusion")

        if d_norm is None and depth_map is not None:
            from .shader_engine import reconstruct_metric_surface
            _, d_norm, d_split = reconstruct_metric_surface(
                depth_map.to(device=device, dtype=torch.float32), H=H, W=W
            )

        if compositing_mode == "Soft Light":
            gain = light_linear * (2.0 ** exposure_ev)
            delta = gain - 1.0
            blend = .5 + .5 * delta / (1 + delta.abs())
            soft = soft_light_blend(img, blend)
            if specular_map is not None and specular_kick > 0:
                spec = specular_map * lighting_strength * specular_kick
                soft = soft + (1 - soft) * (spec / (1 + spec))
            result = img + blend_amount * (soft - img)
            return (
                result,
                contact_sheet(img, light_linear, result, zero, bounce_rgb, zero,
                              linear_to_srgb, luminance_hdr_tonemap, rgb_to_luminance),
                appearance_repair_mask.squeeze(-1),
            )

        amb = max(float(ambient_floor), 0.0)
        effective_fg_gain = (amb + (1.0 - amb) * light_linear) if amb > 0.0 else light_linear
        diff_lum = rgb_to_luminance(effective_fg_gain)
        norm_chroma = effective_fg_gain / torch.clamp(diff_lum, min=1e-5)

        # Legacy explicit far-background protection bypasses scene classification;
        # this retains its documented compositing behavior.
        scene_depth = d_norm if background_protection <= 0 else None
        if scene_depth is not None:
            d_norm = d_norm.to(device=device, dtype=torch.float32)
            if d_norm.shape[1:3] != (H, W):
                d_norm = F.interpolate(
                    d_norm.permute(0, 3, 1, 2).contiguous(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1).contiguous()

            if fg_confidence is not None:
                raw_fg_mask = fg_confidence.to(device=device, dtype=torch.float32)
                if raw_fg_mask.shape[1:3] != (H, W):
                    raw_fg_mask = F.interpolate(
                        raw_fg_mask.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
            else:
                depth_feather = 0.08
                raw_fg_mask = torch.clamp((d_norm - (d_split - depth_feather)) / (2.0 * depth_feather), 0.0, 1.0)
                raw_fg_mask = raw_fg_mask * raw_fg_mask * (3.0 - 2.0 * raw_fg_mask)

            # 4a. Snap Foreground Mask to 1px RGB Silhouette Edges with Anti-Halo Pullback
            m_fg = depth_clamped_rgb_guided_filter(
                raw_fg_mask, img, d_norm, d_split=d_split, radius=5, eps=1e-3
            )

            l_env, c_env_field, m_sky, m_fog_haze, is_studio_cyc = measure_photometric_environment(
                base_linear, d_norm, d_split, m_fg
            )

            # Stage 2C: Normal-Continuous Cast-Shadow Eraser on Background Receivers
            if erase_wall_cast_shadows and normals is not None and float(lighting_strength) > 0.0:
                norm_t = normals.to(device=device, dtype=torch.float32)
                if norm_t.shape[1:3] != (H, W):
                    norm_t = F.interpolate(
                        norm_t.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
                norm_unit = F.normalize(norm_t if norm_t.min() < 0 else norm_t * 2.0 - 1.0, dim=-1)
                dn_x = norm_unit[:, :, 1:, :] - norm_unit[:, :, :-1, :]
                dn_y = norm_unit[:, 1:, :, :] - norm_unit[:, :-1, :, :]
                vx = F.pad(torch.linalg.vector_norm(dn_x, dim=-1, keepdim=True), (0, 0, 0, 1), mode='replicate')
                vy = F.pad(torch.linalg.vector_norm(dn_y, dim=-1, keepdim=True), (0, 0, 0, 0, 0, 1), mode='replicate')
                norm_grad = torch.maximum(vx, vy)
                is_flat_bg = (norm_grad < 0.12) & (m_fg < 0.25) & (m_sky < 0.20)
                bg_lum = rgb_to_luminance(base_linear)
                bg_envelope = _box_filter_bchw(bg_lum.permute(0, 3, 1, 2).contiguous(), 36).permute(0, 2, 3, 1).contiguous()
                shadow_dip = bg_lum / torch.clamp(bg_envelope, min=0.03)
                is_wall_shadow = is_flat_bg & (shadow_dip < 0.70)
                if is_wall_shadow.any():
                    fill_weight = torch.clamp((0.70 - shadow_dip) / 0.50, 0.0, 1.0) * is_wall_shadow.float()
                    base_linear = base_linear + fill_weight * (bg_envelope - bg_lum).clamp_min(0.0) * 0.85
                    appearance_repair_mask = torch.clamp(appearance_repair_mask + fill_weight, 0.0, 1.0)

            bounce_rgb, bounce_confidence, proximity = enclosure_bounce(
                d_norm, m_fg, m_sky, None if strobe is None else strobe * lighting_strength,
                room_bounce_strength, wall_reflectance, enclosure_override,
            )
            effective_fg_gain = effective_fg_gain + bounce_rgb
            diff_lum = rgb_to_luminance(effective_fg_gain)
            norm_chroma = effective_fg_gain / diff_lum.clamp_min(1e-5)

            # 4b. Sub-Black Sanitization, Practical Emitter Protection, & Bidirectional De-Lighting
            base_delit, dark_material_gate, m_emissive = sanitize_subblack_and_delight(
                base_linear, m_fg, r_local=14, r_broad=42, beta=float(crevice_delight), black_gate=black_gate
            )

            # Gate Kelvin chroma_bleed by dark_material_gate so black coats stay neutral jet-black,
            # and apply Von Kries warm-on-warm adaptation so warm skin under <3400K tungsten light
            # does not square its R/B ratio into neon orange ("pumpkin face").
            base_rb = base_delit[..., 0:1] / torch.clamp(base_delit[..., 2:3], min=0.025)
            warm_on_warm = _smoothstep(1.45, 2.40, base_rb) * _smoothstep(1.12, 1.35, norm_chroma[..., 0:1])
            eff_chroma_bleed = float(chroma_bleed) * dark_material_gate * (1.0 - 0.38 * warm_on_warm)
            colorized_fg_gain = (
                (1.0 - eff_chroma_bleed) * diff_lum
                + eff_chroma_bleed * norm_chroma * diff_lum
            )

            # Spatially-varying environmental color wrap (wraps left/right disco neon accurately)
            shadow_weight = torch.clamp(1.0 - (diff_lum / 0.45), 0.0, 1.0) * dark_material_gate
            env_tint = (1.0 - 0.42 * float(chroma_bleed) * shadow_weight) + (
                0.42 * float(chroma_bleed) * shadow_weight * c_env_field
            )
            colorized_fg_gain = colorized_fg_gain * env_tint

            # Protect self-luminous practical props (caveman torch flame, smartphone screen) from being darkened
            colorized_fg_gain = torch.maximum(colorized_fg_gain, m_emissive)

            # 4c. Silhouette Strobe Radiance Transport (Astra / Argon Radiative Transfer)
            # Measure fg_p85 inside the central portrait ROI first (y in 0.12..0.65, x in 0.25..0.75)
            # so bright background extras on the frame edges (Case 2: Techno Club band shirts) never
            # silence additive flash rescue on an underexposed central subject.
            ambient_field = l_env * c_env_field
            strobe_flux = (strobe if strobe is not None else (light_linear - ambient_input).clamp_min(0)) * float(lighting_strength)
            albedo_prior = base_delit.clamp(0.0, 1.0)
            if recovery_mask is not None and shadow_recovery > 0:
                albedo_prior = recover_masked_shadows(base_linear, recovery_mask, float(shadow_recovery), rgb_to_luminance)

            weights = base_delit.new_tensor([0.212639, 0.715169, 0.072192]).view(1, 1, 1, 3)
            delit_lum = (base_delit * weights).sum(-1, keepdim=True)
            ry0, ry1 = int(0.12 * H), max(int(0.12 * H) + 1, int(0.65 * H))
            rx0, rx1 = int(0.25 * W), max(int(0.25 * W) + 1, int(0.75 * W))
            roi_mask_bool = (m_fg[:, ry0:ry1, rx0:rx1, 0] > 0.35)
            fg_mask_bool = (m_fg[..., 0] > 0.35)
            if roi_mask_bool.sum() > 32:
                fg_vals = delit_lum[:, ry0:ry1, rx0:rx1, 0][roi_mask_bool]
                fg_p85 = float(quantile(fg_vals, 0.85).item())
            elif fg_mask_bool.sum() > 16:
                fg_vals = delit_lum[..., 0][fg_mask_bool]
                fg_p85 = float(quantile(fg_vals, 0.85).item())
            else:
                fg_p85 = float(delit_lum.mean().item())
            # Gate additive silhouette rescue so well-exposed subjects (fg_p85 >= 0.22) keep jet-black hair/coats
            subject_is_silhouette = max(0.0, min(1.0, (0.22 - fg_p85) / 0.12))
            eff_illum_ref = torch.maximum(ambient_field, torch.full_like(ambient_field, max(fg_p85, 0.18)))

            noise_floor = 1e-4
            t_sig = ((delit_lum - noise_floor) / (3.0 * noise_floor)).clamp(0.0, 1.0)
            signal_conf = t_sig * t_sig * (3.0 - 2.0 * t_sig)
            illum_valid = (eff_illum_ref.amin(-1, keepdim=True) > noise_floor).to(base_delit.dtype)
            conf = signal_conf * illum_valid
            obs_albedo = (base_delit / eff_illum_ref.clamp_min(noise_floor)).clamp(0.0, 1.0)
            hat_albedo = conf * obs_albedo + (1.0 - conf) * albedo_prior

            # Non-compounding asymmetric photometric composite (Argon Contract v2.1):
            # Preserves strict physical invariants: diffuse light on zero albedo is zero; zero strobe is zero.
            # Shadow side (gain < 1.0): prevents shadow compounding by scaling the ambient floor in pre-existing
            # baked shadow zones (delit_lum < 0.28) up to 0.28, ensuring shadows never compound to dead black.
            # Highlight side (gain >= 1.0) uses energy-conserving soft-knee screen roll-off so highlights wrap smoothly without clipping.
            if float(lighting_strength) <= 0.0:
                fg_lit = torch.zeros_like(base_delit)
            else:
                s_baked_shadow = torch.clamp((0.08 - delit_lum) / 0.06, 0.0, 1.0)
                amb_floor = 0.28 * s_baked_shadow
                eff_floor = amb_floor * min(float(lighting_strength), 1.0)
                shadow_gain = torch.where(
                    colorized_fg_gain > 0.0,
                    eff_floor + (1.0 - eff_floor) * torch.clamp(colorized_fg_gain, 0.0, 1.0),
                    torch.zeros_like(colorized_fg_gain)
                )
                highlight_delta = torch.clamp(colorized_fg_gain - 1.0, min=0.0)
                knee_h = 0.60
                over_h = torch.clamp(highlight_delta - knee_h, min=0.0)
                soft_highlight = base_delit * torch.where(highlight_delta <= knee_h, highlight_delta, knee_h + torch.tanh(over_h))

                fg_lit = base_delit * shadow_gain + soft_highlight

            # 4d. Background Cast Shadow + Softening in Fog/Smoke (`m_fog_haze`)
            if shadow_occ is not None:
                shadow_occ = shadow_occ.to(device=device, dtype=torch.float32)
                if shadow_occ.shape[1:3] != (H, W):
                    shadow_occ = F.interpolate(
                        shadow_occ.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
                shadow_occ = torch.clamp(shadow_occ, 0.0, 1.0)
                # Diffuse hard cast shadows when passing through non-solid fog/smoke
                shadow_blur = _box_filter_bchw(shadow_occ.permute(0, 3, 1, 2).contiguous(), 18).permute(0, 2, 3, 1).contiguous()
                haze_weight = float(atmospheric_haze) * torch.clamp(0.40 + 0.60 * m_fog_haze, 0.0, 1.0)
                shadow_occ = (1.0 - 0.70 * haze_weight) * shadow_occ + (0.70 * haze_weight) * (0.35 * shadow_blur)
            else:
                shadow_occ = torch.zeros_like(m_fg)

            # 4e. 3-Domain Gain Equation + Volumetric Tyndall Scatter in Fog/Haze
            ev_gain = float(2.0 ** float(bg_exposure_ev))
            env_shadow_scale = torch.clamp(0.55 + 0.45 * (l_env / 0.25), 0.55, 1.0)
            strobe_weight = min(float(lighting_strength), 1.0)
            shadow_occ_display = shadow_occ * strobe_weight
            s_eff = float(wall_shadow_density) * env_shadow_scale * strobe_weight

            g_bg_location = ev_gain * (1.0 - s_eff * shadow_occ * (1.0 - m_sky))

            # Optional 3D Volumetric Tyndall Beam Scatter in Fog / Club Haze
            vol_beam = telemetry.get("volumetric_beam") if isinstance(telemetry, dict) else None
            if vol_beam is not None:
                vol_beam = vol_beam.to(device=device, dtype=torch.float32)
                if vol_beam.shape[1:3] != (H, W):
                    vol_beam = F.interpolate(
                        vol_beam.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
                effective_haze = float(atmospheric_haze) * torch.clamp(0.40 + 0.60 * m_fog_haze, 0.0, 1.0)
                tyndall_scatter = vol_beam * float(lighting_strength) * effective_haze * (1.0 - m_sky) * 0.55
            else:
                tyndall_scatter = 0.0

            # Studio Cyclorama / Close-Wall Coupling:
            # When the background is a smooth studio cyclorama (`is_studio_cyc > 0`), transfer
            # the rig's 3D key-light hotspot and full cast shadow from `light_resized` onto the backdrop
            # while flattening Pass 1's pre-baked corner vignette.
            rig_bg_lum = rgb_to_luminance(light_resized)
            unsh_bg = ((m_fg < 0.15) & (shadow_occ < 0.15) & (m_sky < 0.25)).float()
            rig_bg_ref = (rig_bg_lum * unsh_bg).sum(dim=(1, 2), keepdim=True) / torch.clamp(
                unsh_bg.sum(dim=(1, 2), keepdim=True), min=1.0
            )
            rig_cyc_modulation = torch.clamp(light_resized / torch.clamp(rig_bg_ref, min=0.12), 0.08, 1.85)

            # Flatten Pass 1 pre-baked corner vignette on studio cycloramas
            bg_lum_bchw = rgb_to_luminance(base_linear).permute(0, 3, 1, 2).contiguous()
            bg_broad_lum = _box_filter_bchw(bg_lum_bchw, 64).permute(0, 2, 3, 1).contiguous()
            cyc_vignette_flatten = torch.clamp(l_env / torch.clamp(bg_broad_lum, min=0.05), 0.70, 1.45)

            w_cyc = is_studio_cyc * (1.0 - m_sky) * strobe_weight
            base_bg_corrected = base_linear * (1.0 + w_cyc * 0.65 * (cyc_vignette_flatten - 1.0))
            g_bg = (1.0 - w_cyc) * g_bg_location + w_cyc * (ev_gain * rig_cyc_modulation)

            # Part A: Enclosed-Room Additive Wall Illumination (Albedo-modulated by base_bg_corrected so stone columns never bleach white)
            c_geom = float(enclosure_override) if enclosure_override >= 0.0 else (bounce_confidence * proximity)
            wall_signal = _smoothstep(1e-5, 1e-4, rgb_to_luminance(base_linear)) if enclosure_override < 0.0 else 1.0
            delta_L_wall = (
                (1.0 - w_cyc)
                * (1.0 - m_sky)
                * c_geom
                * wall_signal
                * base_bg_corrected
                * (float(wall_reflectance) * 1.15 / math.pi)
                * strobe_flux
                * (1.0 - s_eff * shadow_occ)
            )

            # Direct 3D cavity wall transport from StudioTransportWall (if connected):
            # Modulated by background surface reflectance (base_bg_corrected) + distance proximity gate
            # and dilated foreground exclusion so it illuminates stone/wall texture without milky fog or hair halos!
            if wall_transport is not None:
                if isinstance(wall_transport, dict) and "diffuse" in wall_transport:
                    wt_diff = wall_transport["diffuse"].to(device=device, dtype=torch.float32)
                elif isinstance(wall_transport, torch.Tensor):
                    wt_diff = wall_transport.to(device=device, dtype=torch.float32)
                else:
                    wt_diff = None
                if wt_diff is not None:
                    if wt_diff.shape[1:3] != (H, W):
                        wt_diff = F.interpolate(
                            wt_diff.permute(0, 3, 1, 2).contiguous(),
                            size=(H, W),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1).contiguous()
                    wt_diff = wt_diff.clamp(0.0, 1.5)
                    m_fg_dilated = F.max_pool2d(
                        F.pad(m_fg.permute(0, 3, 1, 2), (3, 3, 3, 3), mode="replicate"),
                        kernel_size=7, stride=1
                    ).permute(0, 2, 3, 1)
                    wall_prox_gate = _smoothstep(0.22, 0.36, d_norm)
                    albedo_wall_transport = (
                        base_bg_corrected
                        * wt_diff
                        * float(wall_reflectance)
                        * 1.25
                        * (1.0 - s_eff * shadow_occ)
                    )
                    delta_L_wall = delta_L_wall + (1.0 - w_cyc) * (1.0 - m_sky) * (1.0 - m_fg_dilated) * wall_prox_gate * albedo_wall_transport

            # Part B: Dielectric Water Surface Light Transport (Case 6: Pool & Case 5: Lake)
            # Cyan/blue water surfaces (where Pass 1 R is near 0.02 due to underwater LED cyan tint)
            # annihilate warm 3000K key lights under pure channel multiplication. Real water reflects overhead
            # directional illumination via dielectric Fresnel surface glints carrying the light's own RGB spectrum.
            r_b = base_linear[..., 0:1]
            g_b = base_linear[..., 1:2]
            b_b = base_linear[..., 2:3]
            is_water_chroma = (
                _smoothstep(1.45, 2.40, b_b / torch.clamp(r_b, min=0.01))
                * _smoothstep(1.15, 1.85, g_b / torch.clamp(r_b, min=0.01))
                * _smoothstep(0.04, 0.16, b_b)
                * (1.0 - m_sky)
                * (1.0 - w_cyc)
                * (1.0 - m_fg)
            )
            water_loc_lum = _box_filter_bchw(bg_lum_bchw, 8).permute(0, 2, 3, 1).contiguous()
            water_ripple = torch.clamp(rgb_to_luminance(base_linear) / torch.clamp(water_loc_lum, min=0.02), 0.65, 1.60)
            full_beam_flux = (light_resized - ambient_input).clamp_min(0.0) * float(lighting_strength)
            delta_L_water = (
                is_water_chroma
                * full_beam_flux
                * (0.20 + 0.22 * (water_ripple - 0.65))
                * (1.0 - s_eff * shadow_occ)
            )

            m_fg_soft = (
                _box_filter_bchw(m_fg.permute(0, 3, 1, 2).contiguous(), 2).permute(0, 2, 3, 1).contiguous()
                if min(H, W) >= 64
                else m_fg
            )
            lit_diffuse = (
                m_fg_soft * fg_lit
                + (1.0 - m_fg_soft) * (base_bg_corrected * g_bg + delta_L_wall + delta_L_water + tyndall_scatter)
            )
        else:
            m_fg = torch.ones_like(diff_lum)
            dark_material_gate = _smoothstep(0.0018, 0.0075, rgb_to_luminance(base_linear)) if black_gate == "Low-signal chroma preservation" else _smoothstep(0.018, 0.070, rgb_to_luminance(base_linear))
            eff_chroma_bleed = float(chroma_bleed) * dark_material_gate
            colorized_fg_gain = (
                (1.0 - eff_chroma_bleed) * diff_lum
                + eff_chroma_bleed * norm_chroma * diff_lum
            )
            lit_diffuse = base_linear * colorized_fg_gain

        # 5. Hue-Preserving Dielectric Specular Catchlights (Protected Against Matte Coat Blobs)
        if float(specular_kick) > 0.0:
            if specular_map is not None:
                spec = specular_map.to(device=device, dtype=torch.float32)
                if spec.shape[1:3] != (H, W):
                    spec = F.interpolate(
                        spec.permute(0, 3, 1, 2).contiguous(),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1).contiguous()
                spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
                # Preserve Kelvin RGB ratios instead of per-channel hard clamping (fixes olive-yellow shift)
                spec_scaled = spec * float(lighting_strength) / SPEC_PASS_REFERENCE
                spec_lum = rgb_to_luminance(spec_scaled)
                spec_lum_clamped = spec_lum / (1.0 + spec_lum) * 1.35
                spec_peak = spec_scaled * (spec_lum_clamped / torch.clamp(spec_lum, min=1e-5))
            else:
                spec_peak = torch.clamp((light_linear - 0.70) / 0.30, 0.0, 1.0)

            # Suppress the 0.12 dielectric floor on matte sub-black wool (where base_linear < 0.025
            # and spec_peak is a broad softbox wash rather than a concentrated glossy catchlight)
            base_lum = rgb_to_luminance(base_linear)
            is_sharp_catchlight = _smoothstep(0.25, 0.65, rgb_to_luminance(spec_peak))
            matte_dark_suppress = torch.maximum(dark_material_gate, is_sharp_catchlight)
            reflectivity_gate = torch.clamp((0.12 * matte_dark_suppress) + base_linear * 1.5, 0.0, 1.0)

            specular_component = spec_peak * float(specular_kick) * 0.35 * reflectivity_gate * m_fg
            lit_hdr = lit_diffuse + specular_component
        else:
            lit_hdr = lit_diffuse

        # Preserve source emission independently of foreground segmentation.
        # Only explicitly marked light sources may bypass studio attenuation.
        source_linear = srgb_to_linear(img)
        lit_hdr = lit_hdr + emission_mask * (source_linear - lit_hdr).clamp_min(0)
        lit_hdr = lit_hdr * (2.0 ** exposure_ev)
        tone = luminance_hdr_tonemap if tone_mapping == "Luminance HDR" else filmic_reinhard_tonemap
        lit_srgb = linear_to_srgb(tone(lit_hdr, white_point=filmic_white_point))
        if background_protection > 0:
            protected = distant_background_weight(depth_map, (H, W)) * background_protection
            lit_srgb = lit_srgb * (1 - protected) + img * protected
        sheet = contact_sheet(img, light_linear + bounce_rgb, lit_srgb, shadow_occ_display,
                              bounce_rgb, bounce_confidence, linear_to_srgb,
                              luminance_hdr_tonemap, rgb_to_luminance)
        return lit_srgb, sheet, appearance_repair_mask.squeeze(-1)


class StudioPhotometricCompositeBeta(StudioPhotometricComposite):
    """Simplified beta UI with the current Linear HDR path fixed as the default.

    The original node remains registered for saved workflows that use its
    compatibility controls. This node exposes the same image-affecting core
    controls while leaving legacy/alternate modes at their established defaults.
    """

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["optional"] = {
            name: value for name, value in schema["optional"].items()
            if name not in {
                "black_gate", "tone_mapping", "exposure_ev",
                "background_protection", "lighting_map_includes_specular",
                "compositing_mode", "blend_amount",
            }
        }
        return schema



NODE_CLASS_MAPPINGS = {
    "StudioPhotometricComposite": StudioPhotometricComposite,
    "StudioPhotometricCompositeBeta": StudioPhotometricCompositeBeta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioPhotometricComposite": "Studio Photometric Composite (Linear HDR)",
    "StudioPhotometricCompositeBeta": "Studio Photometric Composite (Beta Clean)",
}
