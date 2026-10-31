"""Adjust Marigold albedo contribution in linear RGB. Copyright 2026 Maciej Tajber. MIT."""
import math
import torch


def blend_delight(original, albedo, strength):
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("DeLight strength must be between 0 and 1")
    for name, image in (("original", original), ("albedo", albedo)):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError(f"{name} must be a BHWC RGB IMAGE")
        if not image.is_floating_point() or not torch.isfinite(image).all() or (image < 0).any() or (image > 1).any():
            raise ValueError(f"{name} must contain finite sRGB values in [0,1]")
    if original.shape != albedo.shape:
        raise ValueError("Original and albedo must have identical dimensions")
    if strength == 0:
        return original.clone()
    if strength == 1:
        return albedo.to(original.device).clone()
    def linear(x):
        return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).pow(2.4))
    a = linear(original.float())
    b = linear(albedo.to(device=original.device, dtype=torch.float32))
    mixed = torch.lerp(a, b, strength).clamp_min(0)
    return torch.where(mixed <= .0031308, 12.92 * mixed,
                       1.055 * mixed.clamp_min(1e-12).pow(1 / 2.4) - .055).clamp(0, 1)


class StudioLightDeLightStrength:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "original": ("IMAGE",), "albedo": ("IMAGE",),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                    "step": .05, "display": "slider",
                                    "tooltip": "0: original appearance. 1: full Marigold albedo. Intermediate values retain some original lighting."}),
        }}
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("delight_base",)
    FUNCTION = "blend"
    CATEGORY = "StudioLight/Experimental"

    @torch.inference_mode()
    def blend(self, original, albedo, strength):
        return (blend_delight(original, albedo, strength),)


NODE_CLASS_MAPPINGS = {"StudioLightDeLightStrength": StudioLightDeLightStrength}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioLightDeLightStrength": "StudioLight DeLight Strength"}
