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


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """Rec.709 / sRGB Photometric Luminance."""
    weights = torch.tensor([0.212639, 0.715169, 0.072192], device=x.device, dtype=x.dtype).view(1, 1, 1, 3)
    return torch.sum(x * weights, dim=-1, keepdim=True)


def filmic_reinhard_tonemap(x: torch.Tensor, white_point: float = 3.5) -> torch.Tensor:
    """
    Photographic rational shoulder tone reproduction.
    Preserves 100% 1:1 linear contrast and saturation on standard values (<= 0.90),
    gently rolling off specular overshoots and highlights into display headroom without hard-clipping.
    """
    if torch.max(x) <= 0.90:
        return x

    threshold = 0.90
    headroom = 1.0 - threshold  # 0.10
    excess = torch.clamp(x - threshold, min=0.0)

    # Rational shoulder: f(e) = (e * headroom) / (e + headroom)
    # Derivative at excess=0 is exactly 1.0 (seamless C1 continuity with the linear segment).
    # As e -> inf, f(e) -> headroom (0.10), so x_out strictly approaches 1.00 and NEVER clips.
    compressed_excess = (excess * headroom) / (excess + headroom)
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
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lit_image",)
    FUNCTION = "composite"
    CATEGORY = "StudioLight/processing"
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

        # 2. Linearize both plates to scene-referred linear float space
        base_linear = srgb_to_linear(img)
        light_linear = srgb_to_linear(light_resized) * float(lighting_strength)

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
        # Gated by dielectric Fresnel base (0.12) to ensure black patent leather & metals pop
        if float(specular_kick) > 0.0:
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
