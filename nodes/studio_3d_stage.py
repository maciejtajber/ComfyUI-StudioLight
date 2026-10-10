"""
ComfyUI-StudioLight — Studio3DStage (Interactive 3D Stage & Pose Controller)
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Stage 1 of the StudioLight suite:
Manages subject selection (Female/Male mannequins or custom .glb upload),
camera framing, interactive pose editing, and generates baked camera-space
normals and depth maps with intelligent soft-head blurring for prompt hair freedom.
"""

import os
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter


MODEL_PRESETS = [
    "Female Mannequin",
    "Male Mannequin",
    "Custom Upload (.glb)",
]

POSE_PRESETS = [
    "Standing Fashion",
    "American Plan (Hand on Hip)",
    "Sitting Editorial",
    "3/4 Head Turn",
    "Athletic Stance",
    "Crouched Fashion",
]

FRAMING_OPTIONS = [
    "Portrait (Classical Bust)",
    "American Plan (3/4 Shot)",
    "Full Body (Standing Lookbook)",
]


def load_asset_tensor(filename: str) -> Optional[torch.Tensor]:
    """Loads a PNG asset from the custom node assets directory as a float32 torch tensor [1, H, W, 3]."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    asset_path = os.path.join(os.path.dirname(cur_dir), "assets", filename)
    if not os.path.exists(asset_path):
        return None
    try:
        img = Image.open(asset_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)
    except Exception as e:
        print(f"[Studio3DStage] Warning: Failed to load asset {filename}: {e}")
        return None


def apply_soft_head_blur(
    normals: torch.Tensor,
    framing: str,
    blur_radius: float = 25.0,
) -> torch.Tensor:
    """
    Selectively softens normal map boundaries around the head region.
    Prevents the 'baldness leak' in Flux where rigid scalp geometry overrides
    long hair prompts, while preserving sharp collarbones, shoulders, and jawline.
    """
    if blur_radius <= 0.5:
        return normals

    b, h, w, c = normals.shape
    device = normals.device

    # Create spatial head weight mask: 1.0 at top of head, smoothly fading to 0.0 at neck
    y_coords = torch.linspace(0.0, 1.0, h, device=device).view(1, h, 1, 1)

    if "Portrait" in framing:
        # Head occupies roughly top 45% of bust frame
        head_end = 0.45
        neck_transition = 0.15
    elif "American" in framing:
        # Head occupies roughly top 30% of 3/4 frame
        head_end = 0.30
        neck_transition = 0.10
    else:
        # Full body: Head is roughly top 18%
        head_end = 0.18
        neck_transition = 0.08

    # Smooth cosine ease fade
    head_mask = torch.clamp(1.0 - (y_coords - (head_end - neck_transition)) / neck_transition, 0.0, 1.0)
    head_mask = 0.5 - 0.5 * torch.cos(head_mask * torch.pi)

    # Compute Gaussian-blurred normals using ultra-fast 1D separable convolution
    kernel_size = int(blur_radius * 2) | 1  # ensure odd
    sigma = max(blur_radius / 3.0, 0.5)
    k_range = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (k_range / sigma) ** 2)
    gauss_1d = (gauss_1d / gauss_1d.sum()).view(1, 1, 1, kernel_size).to(dtype=normals.dtype)

    kernel_h = gauss_1d.repeat(c, 1, 1, 1)
    kernel_v = gauss_1d.transpose(-1, -2).repeat(c, 1, 1, 1)

    norm_perm = normals.permute(0, 3, 1, 2).contiguous()
    pad = kernel_size // 2
    # Horizontal pass then Vertical pass
    blurred_h = F.conv2d(norm_perm, kernel_h, padding=(0, pad), groups=c)
    blurred_v = F.conv2d(blurred_h, kernel_v, padding=(pad, 0), groups=c)
    blurred_normals = blurred_v.permute(0, 2, 3, 1).contiguous()

    # Re-normalize blurred vector directions
    # Shift [0, 1] -> [-1, 1] -> normalize -> [0, 1]
    vecs = blurred_normals * 2.0 - 1.0
    vec_len = torch.norm(vecs, dim=-1, keepdim=True).clamp(min=1e-5)
    renorm_vecs = (vecs / vec_len) * 0.5 + 0.5

    # Composite: apply blurred normals only in head region
    result = normals * (1.0 - head_mask) + renorm_vecs * head_mask
    return result


class Studio3DStage:
    """
    Studio3DStage — Stage 1 of the ComfyUI-StudioLight Suite.
    Provides an interactive 3D mannequin viewport with Forward Kinematics (FK)
    posing, camera controls, preset models, and outputs camera-space normals + depth
    packaged into a clean `scene_3d` bundle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_preset": (MODEL_PRESETS, {
                    "default": "Female Mannequin",
                }),
                "pose_preset": (POSE_PRESETS, {
                    "default": "Standing Fashion",
                }),
                "framing": (FRAMING_OPTIONS, {
                    "default": "Portrait (Classical Bust)",
                }),
                "soft_head_blur": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON (Hair Freedom)",
                    "label_off": "OFF (Rigid Skull)",
                }),
                "head_blur_radius": ("FLOAT", {
                    "default": 25.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                }),
                "pose_locked": ("BOOLEAN", {
                    "default": False,
                    "label_on": "LOCKED",
                    "label_off": "EDITABLE",
                }),
            },
            "optional": {
                "custom_glb_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "rendered_normals": ("IMAGE",),
                "rendered_depth": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("SCENE_3D", "IMAGE", "IMAGE")
    RETURN_NAMES = ("scene_3d", "normals", "depth")
    FUNCTION = "execute_stage"
    CATEGORY = "StudioLight/Stage"
    DESCRIPTION = (
        "ComfyUI-StudioLight: Stage 1 (3D Mannequin & Pose Controller).\n"
        "• scene_3d: Bundled normal map, depth map, and metadata for StudioLightRig.\n"
        "• normals: Standard IMAGE camera-space normal pass.\n"
        "• depth: Standard IMAGE depth pass."
    )

    def execute_stage(
        self,
        model_preset: str,
        pose_preset: str,
        framing: str,
        soft_head_blur: bool = True,
        head_blur_radius: float = 25.0,
        pose_locked: bool = False,
        custom_glb_path: str = "",
        rendered_normals: Optional[torch.Tensor] = None,
        rendered_depth: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:

        # 1. Resolve Surface Normals
        if rendered_normals is not None:
            normals = rendered_normals.clone()
        else:
            # Fallback to high-definition built-in mannequin assets based on framing
            if "Portrait" in framing:
                asset_name = "mannequin_bust_normal.png"
            elif "American" in framing:
                asset_name = "mannequin_medium_normal.png"
            else:
                asset_name = "mannequin_full_normal.png"

            loaded = load_asset_tensor(asset_name)
            if loaded is not None:
                normals = loaded
            else:
                # Flat fallback normal pointing at camera [0.5, 0.5, 1.0]
                normals = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)
                normals[..., 0] = 0.5
                normals[..., 1] = 0.5
                normals[..., 2] = 1.0

        # 2. Apply Soft Head Blur if requested
        if soft_head_blur:
            normals = apply_soft_head_blur(normals, framing=framing, blur_radius=head_blur_radius)

        # 3. Resolve Linear Depth
        if rendered_depth is not None:
            depth = rendered_depth.clone()
        else:
            # Derive depth approximation from normal Z component
            # Surfaces pointing towards camera are closer (higher depth value)
            nz = normals[..., 2:3]
            depth = torch.clamp((nz - 0.5) * 2.0, 0.0, 1.0)
            depth = depth.repeat(1, 1, 1, 3)

        # 4. Package composite scene_3d dictionary
        scene_3d = {
            "normals": normals,
            "depth": depth,
            "metadata": {
                "model_preset": model_preset,
                "pose_preset": pose_preset,
                "framing": framing,
                "soft_head_blur": soft_head_blur,
                "head_blur_radius": head_blur_radius,
                "pose_locked": pose_locked,
                "custom_glb_path": custom_glb_path,
                "is_stage_asset": True,
                "is_built_in": (rendered_normals is None),
            }
        }

        return (scene_3d, normals, depth)


NODE_CLASS_MAPPINGS = {
    "Studio3DStage": Studio3DStage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Studio3DStage": "Studio 3D Stage (Mannequin & Pose Controller)",
}

# Register ComfyUI Server API route so frontend can fetch 3D meshes directly from assets/
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/studiolight/assets/{filename}")
    async def get_studiolight_asset_route(request):
        filename = request.match_info.get("filename", "")
        safe_filename = os.path.basename(filename)
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        asset_path = os.path.join(os.path.dirname(cur_dir), "assets", safe_filename)
        if os.path.exists(asset_path):
            return web.FileResponse(asset_path)
        return web.Response(status=404, text="Asset not found")
except Exception:
    pass

