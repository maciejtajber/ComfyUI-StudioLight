import torch


def smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    """C^1 continuous Hermite smoothstep polynomial (3t^2 - 2t^3) bounded strictly in [0, 1]."""
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-7), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bounded_s_curve(x: torch.Tensor, contrast: float) -> torch.Tensor:
    """
    Physically bounded sigmoidal S-curve contrast transfer.
    Smoothly pivots around middle grey (0.5) with zero clipping at 0.0 (pure black)
    or 1.0 (pure white), eliminating harsh linear contrast clipping.
    """
    if abs(contrast - 1.0) < 1e-4:
        return x
    c = max(float(contrast), 0.05)
    x_clamped = torch.clamp(x, 0.0, 1.0)
    low = 0.5 * torch.pow(torch.clamp(2.0 * x_clamped, 0.0, 1.0), c)
    high = 1.0 - 0.5 * torch.pow(torch.clamp(2.0 * (1.0 - x_clamped), 0.0, 1.0), c)
    return torch.where(x_clamped < 0.5, low, high)


class SIShadowHighlight:
    """
    Image-space tonal sculptor & photometric preconditioning stage for StudioLight pipeline.
    Allows independent control over shadow depth, highlight specular punch, 
    tonal width, and midtone contrast before VAE encoding.
    Features Hermite C^1 smoothstep tonal isolation (zero midtone bleed)
    and a bounded sigmoidal S-curve contrast transfer.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "shadow_amount": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "highlight_amount": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "shadow_tonal_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "highlight_tonal_width": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}),
                "midtone_contrast": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "adjust"
    CATEGORY = "StudioLight/Photometric"
    DESCRIPTION = (
        "Image-space tonal sculptor & photometric preconditioning stage.\n"
        "Features Hermite C^1 smoothstep tonal isolation (zero midtone bleed) "
        "and bounded sigmoidal S-curve contrast before VAE encoding."
    )

    def adjust(self, image, shadow_amount, highlight_amount, shadow_tonal_width, highlight_tonal_width, midtone_contrast):
        img = image.clone()
        # Luminance L (Rec.709) for shadows, and peak channel intensity for highlights
        lum = 0.2126 * img[..., 0:1] + 0.7152 * img[..., 1:2] + 0.0722 * img[..., 2:3]
        val = torch.max(img, dim=-1, keepdim=True)[0]
        
        # 1. Shadow isolation threshold:
        # Nominal knee at 0.35 when width=0.5, adjustable via shadow_tonal_width.
        # Strict 0.0 above threshold (zero midtone bleed).
        t_shadow = max(min(0.35 * (float(shadow_tonal_width) / 0.5), 0.50), 0.10)
        shadow_mask = 1.0 - smoothstep(0.0, t_shadow, lum)

        # 2. Highlight isolation threshold:
        # Nominal knee at 0.65 when width=0.5, adjustable via highlight_tonal_width.
        # Evaluated on peak specular/channel energy with strict 0.0 below threshold (zero midtone bleed).
        t_highlight = min(max(1.0 - 0.35 * (float(highlight_tonal_width) / 0.5), 0.50), 0.90)
        highlight_mask = smoothstep(t_highlight, 1.0, val)

        # 3. Multiplicative luminance gain with zero midtone crossover:
        # Preserves 100% chromaticity & hue ratio.
        # shadow_amount > 0 lifts shadows; < 0 deepens/crushes shadows
        # highlight_amount > 0 boosts specular punch; < 0 pulls down/compresses highlights
        gain = 1.0 + shadow_mask * float(shadow_amount) + highlight_mask * float(highlight_amount)
        img = img * torch.clamp(gain, min=0.0)

        # 4. Bounded sigmoidal S-curve contrast (no hard linear clipping)
        if abs(float(midtone_contrast) - 1.0) > 1e-4:
            img = bounded_s_curve(img, float(midtone_contrast))

        img = torch.clamp(img, 0.0, 1.0)
        return (img,)


NODE_CLASS_MAPPINGS = {
    "SIShadowHighlight": SIShadowHighlight
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SIShadowHighlight": "SI Shadows & Highlights"
}
