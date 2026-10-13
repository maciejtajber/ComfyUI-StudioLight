"""
ComfyUI-StudioLight — Photometric Composite Node
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Transfers physical 3D studio lighting maps onto draft images in scene-referred
linear Rec.709 HDR space with Kelvin chromaticity preservation and filmic tone reproduction.
"""

import torch
import torch.nn.functional as F


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Accurate IEC 61966-2-1 sRGB EOTF (Linear Rec.709)."""
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow(torch.clamp((x + 0.055) / 1.055, min=1e-7), 2.4)
    )


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """Accurate IEC 61966-2-1 sRGB OETF."""
    x_clamped = torch.clamp(x, 0.0, 1.0)
    return torch.where(
        x_clamped <= 0.0031308,
        x_clamped * 12.92,
        1.055 * torch.pow(torch.clamp(x_clamped, min=1e-7), 1.0 / 2.4) - 0.055
    )


# Specular-pass radiance that maps to a full-strength catchlight (hard modifiers at ~1.5 intensity peak
# at 0.25-0.37, Octabox ~0.10, so soft sources give proportionally gentler catchlights).
SPEC_PASS_REFERENCE = 0.25


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """Rec.709 / sRGB Photometric Luminance."""
    weights = torch.tensor([0.212639, 0.715169, 0.072192], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    return torch.sum(x * weights, dim=-1, keepdim=True)


def filmic_reinhard_tonemap(x: torch.Tensor, white_point: float = 3.5) -> torch.Tensor:
    """
    Photographic rational shoulder tone reproduction.
    Preserves 100% 1:1 linear contrast and saturation on standard values (<= 0.90),
    gently rolling off specular overshoots and highlights into display headroom without hard-clipping.
    Stiffness of shoulder is governed by filmic_white_point.
    """
    threshold = 0.90
    headroom = 1.0 - threshold  # 0.10
    excess = torch.clamp(x - threshold, min=0.0)

    # Quadratic rational shoulder with C1 continuity at threshold
    # f(0) = 0, f'(0) = 1.0, f(e) strictly < headroom for all e >= 0
    # Higher white_point values keep highlights punchier before asymptotic roll-off
    we = max(float(white_point) - threshold, 0.2)
    term = excess / we
    compressed_excess = headroom * (excess * (1.0 + term)) / (excess * (1.0 + term) + headroom)
    return torch.where(x > threshold, threshold + compressed_excess, x)


def luminance_hdr_tonemap(x: torch.Tensor, white_point: float = 3.5) -> torch.Tensor:
    """Continuous luminance shoulder plus luminance-preserving display gamut compression.

    Midtones below 0.60 remain linear. HDR RGB ratios are preserved until display
    gamut compression is required; out-of-gamut chroma moves toward neutral at
    the mapped luminance instead of clipping each RGB channel independently.
    """
    x = torch.clamp(x, min=0.0)
    y = rgb_to_luminance(x)
    threshold, headroom = 0.60, 0.40
    excess = torch.clamp(y - threshold, min=0.0)
    term = excess / max(float(white_point) - threshold, 0.2)
    shoulder = excess * (1.0 + term)
    mapped_y = torch.where(y > threshold, threshold + headroom * shoulder / (shoulder + headroom), y)
    mapped = x * (mapped_y / torch.clamp(y, min=1e-8))
    delta = mapped - mapped_y
    upper = torch.where(delta > 0.0, (1.0 - mapped_y) / torch.clamp(delta, min=1e-8), torch.ones_like(delta))
    lower = torch.where(delta < 0.0, mapped_y / torch.clamp(-delta, min=1e-8), torch.ones_like(delta))
    chroma_scale = torch.minimum(upper, lower).amin(dim=-1, keepdim=True).clamp(0.0, 1.0)
    return mapped_y + delta * chroma_scale


def distant_background_weight(depth: torch.Tensor, size) -> torch.Tensor:
    """Artistic far-background protection, not a sky classifier or validity mask.

    Near-is-white relative depth: protect the farthest 5%, smoothly fade to zero
    at 15%. Normalize independently per image. Constant-depth plates get no
    protection so a flat studio wall is not silently excluded from relighting.
    """
    if depth.ndim != 4 or depth.shape[-1] not in (1, 3):
        raise ValueError('StudioLight background depth must be [B,H,W,1 or 3].')
    if not bool(torch.isfinite(depth).all()):
        raise ValueError('StudioLight background depth must be finite.')
    d = depth[..., :1]
    if d.shape[1:3] != size:
        d = F.interpolate(d.permute(0, 3, 1, 2), size=size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    lo, hi = d.amin(dim=(1, 2), keepdim=True), d.amax(dim=(1, 2), keepdim=True)
    span = hi - lo
    normalized = (d - lo) / span.clamp(min=1e-6)
    t = ((normalized - 0.05) / 0.10).clamp(0.0, 1.0)
    return torch.where(span > 1e-6, 1.0 - t * t * (3.0 - 2.0 * t), torch.zeros_like(t))


class StudioPhotometricComposite:
    """
    Composites 3D deferred relighting maps onto base/draft images in physical linear space.
    Essential for Mid-Stream Latent Intercept and 2-Stage Refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lighting_map": ("IMAGE",),
                "lighting_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.5,
                    "step": 0.05,
                }),
                "ambient_floor": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.50,
                    "step": 0.01,
                    "tooltip": "Scene-referred Linear Rec.709 floor (default 0.00 so StudioLightRig.ambient_intensity controls true shadow density without double-flooring; note 0.08 linear encodes to ~0.313 sRGB).",
                }),
                "specular_kick": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                }),
                "chroma_bleed": ("FLOAT", {
                    "default": 0.40,
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
                "exposure_ev": ("FLOAT", {"default": 0.0, "min": -8.0, "max": 8.0, "step": 0.25,
                    "tooltip": "Exposure in stops before tone mapping. Reduce for close, powerful lights; -2 EV is one quarter of the energy."}),
                "tone_mapping": (["Legacy per-channel", "Luminance HDR"], {"default": "Legacy per-channel"}),
                "background_protection": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Preserve draft pixels in the farthest relative-depth region. Artistic background protection, not sky detection; requires near-is-white depth_map."}),
                "depth_map": ("IMAGE",),
                # Isolated Blinn-Phong lobes from StudioLightRig.specular_pass. When connected, catchlights are
                # confined to the real specular lobes instead of any region where the light map exceeds 0.70.
                "specular_map": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lit_image",)
    FUNCTION = "composite"
    CATEGORY = "StudioLight/Photometric"
    DESCRIPTION = (
        "Composites 3D deferred physical lighting maps onto base/draft images in linear Rec.709 HDR space.\n"
        "Features Kelvin chromaticity transfer, dielectric specular catchlights, and filmic Reinhard tone compression."
    )

    def composite(
        self,
        image: torch.Tensor,
        lighting_map: torch.Tensor,
        lighting_strength: float = 1.0,
        ambient_floor: float = 0.0,
        specular_kick: float = 0.0,
        chroma_bleed: float = 0.40,
        filmic_white_point: float = 3.5,
        specular_map=None,
        exposure_ev: float = 0.0,
        tone_mapping: str = "Legacy per-channel",
        background_protection: float = 0.0,
        depth_map=None,
    ):
        batches = {"image": image.shape[0], "lighting_map": lighting_map.shape[0]}
        if depth_map is not None:
            batches["depth_map"] = depth_map.shape[0]
        if specular_map is not None:
            batches["specular_map"] = specular_map.shape[0]
        if any(count < 1 for count in batches.values()) or len({count for count in batches.values() if count != 1}) > 1:
            raise ValueError(f"StudioLight composite: image/map batches must match or be 1; got {batches}.")
        device = image.device
        img = image.to(device=device, dtype=torch.float32)
        light = lighting_map.to(device=device, dtype=torch.float32)

        # 1. Match spatial dimensions if needed
        if img.shape[1:3] != light.shape[1:3]:
            light_perm = light.permute(0, 3, 1, 2).contiguous()
            light_resized = F.interpolate(
                light_perm,
                size=(img.shape[1], img.shape[2]),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1).contiguous()
        else:
            light_resized = light

        # 2. Linearize Base plate (IEC 61966-2-1 sRGB -> Linear Rec.709);
        #    lighting_map is already scene-referred linear radiance from StudioLightRig (including ambient_intensity).
        base_linear = srgb_to_linear(img)
        light_linear = light_resized * float(lighting_strength)

        # 3. Dynamic Linear Ambient Modulation (default ambient_floor = 0.00 prevents double-flooring
        #    StudioLightRig's ambient_intensity; in Linear Rec.709, e.g. 0.08 lifts black to ~0.313 sRGB).
        amb = max(float(ambient_floor), 0.0)
        if amb > 0.0:
            effective_gain = amb + (1.0 - amb) * light_linear
        else:
            effective_gain = light_linear

        # 4. Luminance-Governed Kelvin Transfer
        diff_lum = rgb_to_luminance(effective_gain)
        norm_chroma = effective_gain / torch.clamp(diff_lum, min=1e-5)
        colorized_gain = (1.0 - chroma_bleed) * diff_lum + chroma_bleed * norm_chroma * diff_lum

        # 5. Modulate Diffuse Base with physical light field
        lit_diffuse = base_linear * colorized_gain

        # 6. Extract Specular Highlights & Add Dielectric Specular Catchlights
        # Reflectivity gate: 0.12 floor (artistic, ~3x real dielectric F0 of 0.04) + albedo term.
        if float(specular_kick) > 0.0:
            if specular_map is not None:
                # Lobe-accurate path: peak comes from the rig's isolated specular pass (Kelvin-tinted, per channel).
                spec = specular_map.to(device=device, dtype=torch.float32)
                if spec.shape[1:3] != img.shape[1:3]:
                    spec = F.interpolate(spec.permute(0, 3, 1, 2).contiguous(), size=(img.shape[1], img.shape[2]),
                                         mode="bilinear", align_corners=False).permute(0, 2, 3, 1).contiguous()
                spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
                spec_peak = torch.clamp(spec * float(lighting_strength) / SPEC_PASS_REFERENCE, 0.0, 1.0)
            else:
                # Legacy path: threshold the summed light map. Veils any broad bright region, not just lobes.
                spec_peak = torch.clamp((light_linear - 0.70) / 0.30, 0.0, 1.0)
            reflectivity_gate = torch.clamp(0.12 + base_linear * 1.5, 0.0, 1.0)
            specular_component = spec_peak * float(specular_kick) * 0.35 * reflectivity_gate
            lit_hdr = lit_diffuse + specular_component
        else:
            lit_hdr = lit_diffuse

        # Explicit exposure retains scene-referred HDR until the display transform.
        lit_hdr = lit_hdr * (2.0 ** float(exposure_ev))
        if tone_mapping == "Luminance HDR":
            lit_toned = luminance_hdr_tonemap(lit_hdr, white_point=float(filmic_white_point))
        elif tone_mapping == "Legacy per-channel":
            lit_toned = filmic_reinhard_tonemap(lit_hdr, white_point=float(filmic_white_point))
        else:
            raise ValueError(f"Unknown StudioLight tone mapping: {tone_mapping}")

        # 8. Encode back to display sRGB
        lit_srgb = linear_to_srgb(lit_toned)
        protection = min(max(float(background_protection), 0.0), 1.0)
        if protection > 0.0:
            if depth_map is None:
                raise ValueError('StudioLight background protection requires a depth_map.')
            weight = distant_background_weight(depth_map.to(device=device, dtype=torch.float32), img.shape[1:3]) * protection
            lit_srgb = lit_srgb * (1.0 - weight) + img * weight
        return (lit_srgb,)


NODE_CLASS_MAPPINGS = {
    "StudioPhotometricComposite": StudioPhotometricComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioPhotometricComposite": "Studio Photometric Composite (Linear HDR)",
}
