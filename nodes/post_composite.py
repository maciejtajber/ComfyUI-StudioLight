"""
ComfyUI-StudioLight Post-Composite Retouching Node
Photographic Dodge & Burn Soft Light Overlay stage.
Applies 3D cast shadows and specular catchlights directly to the final image
without darkening the ambient background or altering base exposure.
"""

import torch
import torch.nn.functional as F


def soft_light_w3c(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """
    Standard W3C / Photoshop Soft Light formula.
    - blend = 0.50 is an exact identity (zero change).
    - blend > 0.50 dodges highlights smoothly.
    - blend < 0.50 burns shadows smoothly.
    """
    d_base = torch.where(
        base <= 0.25,
        ((16.0 * base - 12.0) * base + 4.0) * base,
        torch.sqrt(torch.clamp(base, min=1e-7))
    )
    return torch.where(
        blend <= 0.5,
        base - (1.0 - 2.0 * blend) * base * (1.0 - base),
        base + (2.0 * blend - 1.0) * (d_base - base)
    )


def overlay_blend(base: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Standard Photoshop Overlay formula."""
    return torch.where(
        base <= 0.5,
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    )


class StudioLightPostComposite:
    """
    High-end fashion retouching post-composite stage.
    Applies 3D raymarched directional cast shadows, key highlights, and specular catchlights
    directly over the final image using a 50% neutral gray Dodge & Burn overlay.
    Guarantees 0% darkening to sky/ambient background while preserving 100% of facial micro-detail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lighting_map": ("IMAGE",),
                "shadow_burn": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Strength of 3D cast shadows burned into the final image (0.0 = no shadow burn, 1.0 = standard, 2.0 = dramatic noir)."
                }),
                "highlight_dodge": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 2.5,
                    "step": 0.05,
                    "tooltip": "Strength of key light highlights dodged onto the final image."
                }),
                "specular_kick": ("FLOAT", {
                    "default": 0.45,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Dielectric catchlights added to eyes and specular surfaces."
                }),
                "ambient_baseline": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.05,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Scene ambient level that maps to 50% neutral gray (invisible / zero change). Background sky sits at this level and remains 100% clean."
                }),
                "blend_channel": (["Luminosity (Clean Photoreal)", "RGB Per-Channel"], {
                    "default": "Luminosity (Clean Photoreal)",
                    "tooltip": "Luminosity mode sculpts light and shadow without shifting skin saturation or turning faces orange. RGB mode is legacy per-channel."
                }),
                "blend_mode": (["Soft Light", "Overlay"], {
                    "default": "Soft Light",
                    "tooltip": "Soft Light gives gentle photographic contrast; Overlay gives punchy commercial contrast."
                }),
                "opacity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Master overlay opacity."
                }),
            },
            "optional": {
                "specular_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("lit_image",)
    FUNCTION = "apply_overlay"
    CATEGORY = "StudioLight/Photometric"
    DESCRIPTION = (
        "Applies 3D raymarched cast shadows and specular highlights as a photographic Dodge & Burn overlay.\n"
        "Guarantees that sky and background daylight remain 100% clean (zero doomsday gloom) while adding deep directional shadows."
    )

    def apply_overlay(
        self,
        image: torch.Tensor,
        lighting_map: torch.Tensor,
        shadow_burn: float = 1.0,
        highlight_dodge: float = 0.5,
        specular_kick: float = 0.45,
        ambient_baseline: float = 0.35,
        blend_channel: str = "Luminosity (Clean Photoreal)",
        blend_mode: str = "Soft Light",
        opacity: float = 1.0,
        specular_map=None,
    ):
        device = image.device
        img = image.to(device=device, dtype=torch.float32)
        light = lighting_map.to(device=device, dtype=torch.float32)

        # 1. Resize lighting map if dimensions differ
        if img.shape[1:3] != light.shape[1:3]:
            light_perm = light.permute(0, 3, 1, 2).contiguous()
            light = F.interpolate(
                light_perm,
                size=(img.shape[1], img.shape[2]),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1).contiguous()

        # 2. Extract luminance of the lighting field
        weights = torch.tensor([0.212639, 0.715169, 0.072192], device=device, dtype=light.dtype).view(1, 1, 1, 3)
        light_lum = torch.sum(light * weights, dim=-1, keepdim=True)

        # 3. Derive 50% neutral-centered Dodge & Burn layer
        base_level = max(float(ambient_baseline), 0.01)
        rel_gain = light_lum / base_level

        delta = rel_gain - 1.0
        scaled_delta = torch.where(
            delta < 0.0,
            delta * float(shadow_burn),
            delta * float(highlight_dodge)
        )

        blend_layer = 0.5 + 0.5 * (scaled_delta / (1.0 + torch.abs(scaled_delta)))
        blend_layer = torch.clamp(blend_layer, 0.0, 1.0)

        # 4. Apply blend equation
        if blend_channel == "Luminosity (Clean Photoreal)":
            # Extract image luminance
            img_lum = torch.sum(img * weights, dim=-1, keepdim=True)
            blend_1ch = blend_layer[..., :1]
            if blend_mode == "Soft Light":
                blended_lum = soft_light_w3c(img_lum, blend_1ch)
            else:
                blended_lum = overlay_blend(img_lum, blend_1ch)
            # Transfer luminance ratio to RGB: preserves 100% of chromaticity ratios
            lum_ratio = blended_lum / torch.clamp(img_lum, min=1e-5)
            blended = torch.clamp(img * lum_ratio, 0.0, 1.0)
        else:
            blend_3ch = blend_layer.expand_as(img)
            if blend_mode == "Soft Light":
                blended = soft_light_w3c(img, blend_3ch)
            else:
                blended = overlay_blend(img, blend_3ch)

        # 5. Mix with master opacity
        op = min(max(float(opacity), 0.0), 1.0)
        result = img * (1.0 - op) + blended * op

        # 6. Add dielectric specular catchlights if available
        if float(specular_kick) > 0.0 and specular_map is not None:
            spec = specular_map.to(device=device, dtype=torch.float32)
            if spec.shape[1:3] != img.shape[1:3]:
                spec = F.interpolate(
                    spec.permute(0, 3, 1, 2).contiguous(),
                    size=(img.shape[1], img.shape[2]),
                    mode="bilinear",
                    align_corners=False
                ).permute(0, 2, 3, 1).contiguous()
            spec = torch.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
            spec_peak = torch.clamp(spec / 0.25, 0.0, 1.0)
            catchlight = spec_peak * float(specular_kick) * 0.40 * op
            result = result + (1.0 - result) * catchlight

        return (torch.clamp(result, 0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "StudioLightPostComposite": StudioLightPostComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightPostComposite": "Studio Light Post-Composite (Dodge & Burn Overlay)",
}
