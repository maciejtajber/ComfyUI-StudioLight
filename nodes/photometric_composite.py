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
    ):
        device = image.device
        img = image.to(device=device, dtype=torch.float32)
        light = lighting_map.to(device=device, dtype=torch.float32)

        # 1. Match spatial dimensions if needed
        if img.shape[1:3] != light.shape[1:3]:
            light_perm = light.permute(0, 3, 1, 2)
            light_resized = F.interpolate(
                light_perm,
                size=(img.shape[1], img.shape[2]),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)
        else:
            light_resized = light

        # 2. Linearize Base plate; Lighting map is already pure linear physical radiance from StudioLightRig
        base_linear = srgb_to_linear(img)
        light_linear = light_resized * float(lighting_strength)

        # 3. Dynamic Ambient Modulation (Zero double-flooring: physical ambient already exists in lighting_map)
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
                    spec = F.interpolate(spec.permute(0, 3, 1, 2), size=(img.shape[1], img.shape[2]),
                                         mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
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

        # 7. Filmic Tone Reproduction (Extended Reinhard)
        lit_toned = filmic_reinhard_tonemap(lit_hdr, white_point=float(filmic_white_point))

        # 8. Encode back to display sRGB
        lit_srgb = linear_to_srgb(lit_toned)
        return (lit_srgb,)


NODE_CLASS_MAPPINGS = {
    "StudioPhotometricComposite": StudioPhotometricComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioPhotometricComposite": "Studio Photometric Composite (Linear HDR)",
}
