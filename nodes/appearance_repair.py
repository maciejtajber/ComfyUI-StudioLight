"""
ComfyUI-StudioLight — Stage 3 Appearance Repair Node
Author: Maciej Tajber // METAMATERIAL

Implements Stage 3 of the StudioLight Rebuild:
"Masked Appearance Repair: Inpaint face/rock & erase wall cast shadows BEFORE lighting"

When Stage 2 Preflight detects an under-resolved dark face silhouette, or when
Pass 1 bakes a harsh cast shadow onto a flat background wall/column, this node
restores clean, readable surface appearance and outputs a repair_mask for Pass 2.
"""

import json
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

def _to_bhwc(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(-1)
    if x.ndim == 3:
        if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            return x.permute(1, 2, 0).unsqueeze(0)
        return x.unsqueeze(-1)
    if x.ndim == 4 and x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        return x.permute(0, 2, 3, 1)
    return x

def _compute_gradients_2d(t: torch.Tensor) -> torch.Tensor:
    """Computes 2D spatial gradient magnitude for BHWC tensor."""
    dx = t[:, :, 1:, :] - t[:, :, :-1, :]
    dy = t[:, 1:, :, :] - t[:, :-1, :, :]
    pad_x = F.pad(dx, (0, 0, 0, 1, 0, 0), mode="replicate")
    pad_y = F.pad(dy, (0, 0, 0, 0, 0, 1), mode="replicate")
    return (pad_x.square().sum(dim=-1, keepdim=True) + pad_y.square().sum(dim=-1, keepdim=True)).sqrt()


def delight_crevices_pointwise(img_rgb: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    """
    Strictly pointwise luma lift (ZERO spatial box/pool filters = ZERO square blocks).
    Smooth per-pixel toe lift below luma = 0.20 (no spatial pooling, no channel stripping).
    """
    luma = 0.2126 * img_rgb[..., 0:1] + 0.7152 * img_rgb[..., 1:2] + 0.0722 * img_rgb[..., 2:3]
    
    toe = torch.clamp((0.20 - luma) / 0.20, 0.0, 1.0)
    smooth_toe = toe * toe * (3.0 - 2.0 * toe)
    gain = 1.0 + strength * smooth_toe
    return torch.clamp(img_rgb * gain, 0.0, 1.0)


def delight_crevices_luma_safe(img_rgb: torch.Tensor, strength: float = 0.35) -> torch.Tensor:
    """Backward-compatible alias for delight_crevices_pointwise."""
    return delight_crevices_pointwise(img_rgb, strength=strength)


class StudioAppearanceRepair:
    """
    Stage 3: Masked Appearance Repair & Flat-Wall Cast Shadow Eraser.
    Evaluates Preflight report and conditionally repairs under-resolved face ROIs
    and erases flat-wall cast shadows using surface normal and depth continuity.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "draft": ("IMAGE",),
                "review_report": ("STRING", {"forceInput": True}),
                "repair_mode": (["auto_adaptive_lift", "external_inpaint_blend", "passthrough"],),
                "feather_pixels": ("INT", {"default": 16, "min": 0, "max": 128}),
                "target_face_exposure": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "external_inpaint": ("IMAGE",),
                "manual_mask": ("MASK",),
                "depth": ("IMAGE",),
                "normals": ("IMAGE",),
                "subject_mask": ("MASK",),
                "erase_wall_shadows": ("BOOLEAN", {"default": False, "label_on": "ON (Auto-Erase)", "label_off": "OFF (Clean Pass 1)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("repaired_draft", "repair_mask", "repair_status")
    FUNCTION = "repair"
    CATEGORY = "StudioLight/Transport"

    def repair(
        self,
        draft: torch.Tensor,
        review_report: str,
        repair_mode: str = "auto_adaptive_lift",
        feather_pixels: int = 16,
        target_face_exposure: float = 0.45,
        external_inpaint: Optional[torch.Tensor] = None,
        manual_mask: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        normals: Optional[torch.Tensor] = None,
        subject_mask: Optional[torch.Tensor] = None,
        erase_wall_shadows: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        c = _to_bhwc(draft).float()
        b, h, w, ch = c.shape

        # Parse preflight reports
        try:
            reports = json.loads(review_report)
            if isinstance(reports, dict):
                reports = [reports]
        except Exception:
            reports = [{"status": "unknown", "reasons": [], "roi": [0, 0, w, h]}]

        needs_face_repair = False
        rois = []
        for rep in reports:
            if rep.get("status") == "needs_review_or_repair":
                needs_face_repair = True
                rois.append(rep.get("roi", [0, 0, w, h]))

        # Generate base repair mask
        mask = torch.zeros((b, h, w, 1), dtype=c.dtype, device=c.device)
        if manual_mask is not None:
            m = _to_bhwc(manual_mask).float().to(device=c.device)
            if m.shape[-1] > 1:
                m = m[..., :3].mean(dim=-1, keepdim=True)
            if m.shape[1:3] != (h, w):
                m = F.interpolate(m.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            mask = m.clamp(0.0, 1.0)
            needs_face_repair = mask.max().item() > 0.05
        elif needs_face_repair and rois:
            for roi in rois:
                x0, y0, x1, y1 = roi
                x0, y0 = max(0, int(x0)), max(0, int(y0))
                x1, y1 = min(w, int(x1)), min(h, int(y1))
                mask[:, y0:y1, x0:x1, :] = 1.0

        # Feather face mask if requested
        if feather_pixels > 0 and mask.max().item() > 0:
            k = feather_pixels * 2 + 1
            mask_padded = F.pad(mask.permute(0, 3, 1, 2), (feather_pixels, feather_pixels, feather_pixels, feather_pixels), mode="replicate")
            mask = F.avg_pool2d(mask_padded, k, stride=1).permute(0, 2, 3, 1).clamp(0.0, 1.0)

        repaired = c.clone()
        face_status = "none"

        # 1. Face / Subject Adaptive Tone Repair
        if repair_mode != "passthrough":
            if repair_mode == "external_inpaint_blend" and external_inpaint is not None:
                inp = _to_bhwc(external_inpaint).float().to(device=c.device)
                if inp.shape[1:3] != (h, w):
                    inp = F.interpolate(inp.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                repaired = c * (1.0 - mask) + inp * mask
                face_status = "external_inpaint_blend"
            elif needs_face_repair and (repair_mode == "auto_adaptive_lift" or external_inpaint is None):
                for roi in rois:
                    x0, y0, x1, y1 = roi
                    x0, y0 = max(0, int(x0)), max(0, int(y0))
                    x1, y1 = min(w, int(x1)), min(h, int(y1))
                    if x1 > x0 and y1 > y0:
                        roi_patch = c[:, y0:y1, x0:x1, :]
                        lum = (roi_patch * roi_patch.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1, keepdim=True)
                        cur_exp = max(float(lum.mean().item()), 0.01)
                        boost = min(target_face_exposure / cur_exp, 8.0)
                        lifted_patch = torch.pow(roi_patch.clamp_min(1e-4), 0.6) * (boost ** 0.5)
                        lifted_patch = lifted_patch.clamp(0.0, 1.0)
                        local_mask = mask[:, y0:y1, x0:x1, :]
                        repaired[:, y0:y1, x0:x1, :] = roi_patch * (1.0 - local_mask) + lifted_patch * local_mask
                face_status = "auto_adaptive_lift"

        # 2. Geometry-Continuous Cast-Shadow Eraser on Background Enclosure / Walls
        wall_shadow_erased = False
        wall_erased_pixels = 0
        if erase_wall_shadows and depth is not None and normals is not None:
            d = _to_bhwc(depth).float().to(device=c.device)
            if d.shape[-1] > 1:
                d = d[..., :1]
            if d.shape[1:3] != (h, w):
                d = F.interpolate(d.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            n = _to_bhwc(normals).float().to(device=c.device)
            if n.shape[1:3] != (h, w):
                n = F.interpolate(n.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            # Map [0, 1] encoded normals to unit sphere [-1, 1]
            if n.shape[-1] == 3 and n.max() <= 1.0 and n.min() >= 0.0:
                n_unit = n * 2.0 - 1.0
            else:
                n_unit = n

            # Determine foreground subject mask
            if subject_mask is not None:
                sm = _to_bhwc(subject_mask).float().to(device=c.device)
                if sm.shape[-1] > 1:
                    sm = sm[..., :3].mean(dim=-1, keepdim=True)
                if sm.shape[1:3] != (h, w):
                    sm = F.interpolate(sm.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                m_fg = sm.clamp(0.0, 1.0)
            else:
                # Automatic foreground derivation from depth median and face ROIs
                span = float((d.max() - d.min()).item())
                if span > 0.08:
                    split = float(d.median().item())
                    m_fg = torch.clamp((d - split + 0.05) / 0.10, 0.0, 1.0)
                else:
                    m_fg = torch.zeros_like(d)
                if rois:
                    for roi in rois:
                        x0, y0, x1, y1 = roi
                        m_fg[:, int(y0):int(y1), int(x0):int(x1), :] = 1.0

            # Mid-ground receiver wall/column zone: background pixels (1 - m_fg) with depth > 0.20
            # (Captures columns, walls, rocks; excludes distant night sky/street where depth <= 0.20)
            wall_receiver_zone = (1.0 - m_fg).clamp(0.0, 1.0) * (d > 0.20).float()

            # Normal & Depth spatial gradients
            grad_n = _compute_gradients_2d(n_unit)
            grad_d = _compute_gradients_2d(d)

            # Flat, smooth geometry mask: normal gradient < 0.18 and depth gradient < 0.08
            is_smooth_wall = (wall_receiver_zone > 0.35) & (grad_n < 0.18) & (grad_d < 0.08)

            lum = (c * c.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1, keepdim=True)
            smooth_lum = lum[is_smooth_wall]

            if smooth_lum.numel() > 100:
                # 80th percentile luminance represents clean unshadowed wall surface
                y_lit = float(torch.quantile(smooth_lum.float(), 0.80).item())
                y_threshold = 0.78 * y_lit

                # Cast shadows: smooth geometry with sharp luminance drop
                wall_shadow_pixels = is_smooth_wall & (lum < y_threshold)
                wall_erased_pixels = int(wall_shadow_pixels.sum().item())

                if wall_erased_pixels > 50:
                    w_shadow_mask = wall_shadow_pixels.float()
                    # Feather the detected cast-shadow boundary
                    if feather_pixels > 0:
                        k = feather_pixels * 2 + 1
                        w_pad = F.pad(w_shadow_mask.permute(0, 3, 1, 2), (feather_pixels, feather_pixels, feather_pixels, feather_pixels), mode="replicate")
                        w_shadow_mask = F.avg_pool2d(w_pad, k, stride=1).permute(0, 2, 3, 1).clamp(0.0, 1.0)

                    # Reference clean lit stone color & chromaticity
                    lit_stone_pixels = (is_smooth_wall & (lum >= y_lit * 0.75))[..., 0]
                    if lit_stone_pixels.sum() > 50:
                        ref_color = c[lit_stone_pixels].mean(dim=0, keepdim=True).view(1, 1, 1, 3)
                    else:
                        ref_color = c[is_smooth_wall[..., 0]].mean(dim=0, keepdim=True).view(1, 1, 1, 3)

                    # High-pass micro-texture extraction to preserve stone/rock grain
                    c_pad = F.pad(c.permute(0, 3, 1, 2), (7, 7, 7, 7), mode="replicate")
                    low_pass = F.avg_pool2d(c_pad, 15, stride=1).permute(0, 2, 3, 1)
                    high_pass = c - low_pass

                    # Lifted wall baseline matching unshadowed stone exposure and color
                    scale = (y_lit / (low_pass + 1e-4)).clamp(1.0, 6.0)
                    target_chroma = ref_color / (ref_color.sum(dim=-1, keepdim=True) + 1e-6)
                    lifted_baseline = low_pass * scale * 0.5 + target_chroma * y_lit * 0.5
                    lifted_wall = (lifted_baseline + high_pass).clamp(0.0, 1.0)

                    # Seamlessly blend lifted stone back into repaired draft
                    repaired = repaired * (1.0 - w_shadow_mask) + lifted_wall * w_shadow_mask
                    # Add to composite repair_mask for Pass 2 refinement
                    mask = torch.max(mask, w_shadow_mask)
                    wall_shadow_erased = True

        status = {
            "repaired": needs_face_repair or wall_shadow_erased,
            "face_mode": face_status,
            "wall_shadow_erased": wall_shadow_erased,
            "wall_shadow_pixels": wall_erased_pixels,
            "rois": rois
        }

        return (repaired.clamp(0.0, 1.0), mask[..., 0], json.dumps(status))

NODE_CLASS_MAPPINGS = {
    "StudioAppearanceRepair": StudioAppearanceRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioAppearanceRepair": "StudioLight · Stage 3 Appearance Repair",
}
