import torch

class SIShadowHighlight:
    """
    Dual-threshold Shadow & Highlight adjustment node for StudioLight pipeline.
    Allows independent control over shadow depth, highlight specular punch, 
    tonal width, and midtone contrast before VAE encoding.
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

    def adjust(self, image, shadow_amount, highlight_amount, shadow_tonal_width, highlight_tonal_width, midtone_contrast):
        img = image.clone()
        # Luminance L (Rec.709)
        lum = 0.2126 * img[..., 0:1] + 0.7152 * img[..., 1:2] + 0.0722 * img[..., 2:3]
        
        # Shadow mask (higher weight on dark pixels)
        shadow_mask = torch.pow(torch.clamp(1.0 - lum, 0.0, 1.0), shadow_tonal_width)
        
        # Highlight mask (higher weight on bright pixels)
        highlight_mask = torch.pow(torch.clamp(lum, 0.0, 1.0), highlight_tonal_width)
        
        # Multiplicative luminance gain preserves 100% color fidelity and saturation
        # shadow_amount > 0 lifts shadows; < 0 deepens/crushes shadows
        # highlight_amount > 0 boosts specular kicks; < 0 dims highlights
        gain = 1.0 + shadow_mask * shadow_amount + highlight_mask * highlight_amount
        img = img * torch.clamp(gain, min=0.0)
        
        # Midtone contrast
        if midtone_contrast != 1.0:
            img = (img - 0.5) * midtone_contrast + 0.5
            
        img = torch.clamp(img, 0.0, 1.0)
        return (img,)

NODE_CLASS_MAPPINGS = {
    "SIShadowHighlight": SIShadowHighlight
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SIShadowHighlight": "SI Shadows & Highlights"
}
