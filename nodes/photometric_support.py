"""Validation and conservative, explicitly approximate relighting support.

All lighting fields are linear RGB gains, not calibrated radiometric units.
Relative depth is near=1. No operation in this module assigns metres to it.
"""
import math
from functools import lru_cache
import torch
import torch.nn.functional as F


def image_tensor(value, name, device, channels=(3,)):
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a BHWC tensor")
    if min(value.shape[:3]) < 1:
        raise ValueError(f"{name}: batches and spatial dimensions must be nonempty")
    if value.shape[-1] not in channels:
        raise ValueError(f"{name}: expected channels {channels}, got {value.shape[-1]}")
    value = value.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def resize_image(value, size):
    if value.shape[1:3] == size:
        return value
    return F.interpolate(value.permute(0, 3, 1, 2), size=size,
                         mode='bilinear', align_corners=False).permute(0, 2, 3, 1)


def scalar(value, name, low, high):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return value


def quantile(value, q):
    """CPU fallback for MPS order statistics; CUDA/CPU keep their device."""
    return torch.quantile(value.cpu() if value.device.type == 'mps' else value, q).to(value.device)


def soft_light_blend(base, blend):
    """W3C Soft Light in display RGB; blend=0.5 is identity."""
    d = torch.where(base <= .25, ((16 * base - 12) * base + 4) * base,
                    torch.sqrt(base.clamp_min(0)))
    return torch.where(blend <= .5, base - (1 - 2 * blend) * base * (1 - base),
                       base + (2 * blend - 1) * (d - base))


def distant_background_weight(depth, size):
    """Legacy protection mask: near=1 relative depth, independently normalized."""
    depth = image_tensor(depth, 'depth_map', depth.device, (1, 3))[..., :1]
    lo = depth.amin((1, 2, 3), keepdim=True)
    span = depth.amax((1, 2, 3), keepdim=True) - lo
    normalized = (depth - lo) / span.clamp_min(1e-6)
    mask = ((normalized <= .05 + 1e-6) & (span > 1e-6)).float()
    return resize_image(mask, size)


def enclosure_bounce(d_norm, foreground, sky, strobe_diffuse, strength,
                     wall_reflectance, override=-1.):
    """Per-image RGB fill heuristic, never a metric room or GI estimator.

    Only visible non-sky background samples receive the strobe. Their RGB
    diffuse gain drives additive fill. No evidence or no strobe => no fill.
    Returns bounce RGB, enclosure confidence, and proximity score.
    """
    zero = torch.zeros_like(foreground)
    if d_norm is None or strobe_diffuse is None:
        return zero.expand(-1, -1, -1, 3), zero, zero
    # These statistics describe one frame. Never pool pixels from different
    # batch items, or one scene can change another scene's bounce estimate.
    if foreground.shape[0] > 1:
        per_image = [
            enclosure_bounce(
                d_norm[i:i + 1], foreground[i:i + 1], sky[i:i + 1],
                strobe_diffuse[i:i + 1], strength, wall_reflectance, override,
            )
            for i in range(foreground.shape[0])
        ]
        return tuple(torch.cat([item[j] for item in per_image], dim=0) for j in range(3))
    fg = foreground[..., 0] > .8
    wall = (foreground[..., 0] < .15) & (sky[..., 0] < .25) & (d_norm[..., 0] > .05)
    min_count = max(4, math.ceil(d_norm.shape[1] * d_norm.shape[2] * .01))
    if int(fg.sum()) < min_count or int(wall.sum()) < min_count:
        return zero.expand(-1, -1, -1, 3), zero, zero
    fg_depth = d_norm[..., 0][fg]
    wall_depth = d_norm[..., 0][wall]
    gap = torch.median(fg_depth) - torch.median(wall_depth)
    span = d_norm.max() - d_norm.min()
    if float(span) < 1e-4 or float(gap) <= .02:
        return zero.expand(-1, -1, -1, 3), zero, zero
    # Relative-depth tuning, NOT metres. Confidence expresses image support only.
    proximity = ((.60 - gap) / .45).clamp(0, 1) if override < 0 else gap.new_tensor(override)
    coverage = (wall.float().mean() / .25).clamp(0, 1)
    confidence = coverage * (fg.float().mean() / .05).clamp(0, 1)
    mean_rgb = (strobe_diffuse * wall.unsqueeze(-1)).sum((1, 2), keepdim=True) / wall.sum().clamp_min(1)
    bounce = mean_rgb * wall_reflectance * strength * proximity * confidence
    return bounce.expand_as(strobe_diffuse), zero + confidence, zero + proximity


def recover_masked_shadows(base, mask, amount, luminance):
    """Optional material prior from exposed pixels in a user-selected material.

    Does not assign a skin colour or use the new lighting to infer reflectance.
    Insufficient reference pixels => unchanged. Mask should select ONE material.
    """
    if mask is None or amount <= 0:
        return base
    # The trusted material colour is a per-image prior. A single pooled
    # median would transfer colour between unrelated images in a batch.
    if base.shape[0] > 1:
        return torch.cat([
            recover_masked_shadows(base[i:i + 1], mask[i:i + 1], amount, luminance)
            for i in range(base.shape[0])
        ], dim=0)
    lum = luminance(base)
    trusted = (mask[..., 0] > .5) & (lum[..., 0] >= .015) & (lum[..., 0] <= .4)
    if int(trusted.sum()) < 4:
        return base
    reference = base[trusted].median(dim=0).values.view(1, 1, 1, 3)
    # Cap reconstructed material to 0.05 linear luminance; lift only dark zones.
    reference = reference * (0.05 / luminance(reference).clamp_min(1e-6)).clamp_max(1)
    weight = mask * ((.015 - lum) / .015).clamp(0, 1) * amount
    return base + weight * (reference - base).clamp_min(0)


@lru_cache(maxsize=16)
def _label_pixels(width, text):
    from PIL import Image, ImageDraw
    import numpy as np
    label = Image.new('RGB', (width, 20), (20, 20, 20))
    ImageDraw.Draw(label).text((3, 3), text, fill=(235, 235, 235))
    return torch.from_numpy(np.asarray(label).copy()).float().unsqueeze(0) / 255


def contact_sheet(image, light, result, shadow, bounce, confidence, encode, tonemap, luminance):
    h, w = max(1, image.shape[1] // 2), max(1, image.shape[2] // 2)
    # False-colour diagnostics use numeric display values intentionally.
    bounce_lum = luminance(bounce)
    telemetry = torch.cat((shadow.clamp(0, 1), bounce_lum / (1 + bounce_lum), confidence.clamp(0, 1)), -1)
    panels = (image, encode(tonemap(light)), result, telemetry)
    labels = ('Pass 1 / sRGB', 'Light / tone mapped', 'Lit proxy / sRGB', 'R shadow | G bounce | B confidence')
    tiles = []
    for panel, label in zip(panels, labels):
        pixels = resize_image(panel, (h, w)).clamp(0, 1)
        header = _label_pixels(w, label).to(device=image.device, dtype=image.dtype)
        tiles.append(torch.cat((header, pixels), dim=1))
    return torch.cat((torch.cat(tiles[:2], 2), torch.cat(tiles[2:], 2)), 1)
