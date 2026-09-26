"""
ComfyUI-StudioLight — Professional Photographic Studio Lighting Controller
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Flagship 3-point photographic lighting console node for ComfyUI.
Integrates physics-based deferred normal relighting, high-signal photographic
prompt synthesis, step-scheduled CLIP conditioning, and visual VFX probe previews.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
from .color_utils import MODIFIER_PRESETS
from .preset_manager import list_presets, load_preset_data
from .shader_engine import (
    render_studio_lighting,
    compute_light_shading,
    azimuth_elevation_to_vector,
    kelvin_to_rgb,
)
from .prompt_synthesizer import synthesize_high_signal_prompt, encode_scheduled_conditioning


FRAMING_OPTIONS = [
    "Portrait (Classical Bust)",
    "American Plan (3/4 Shot)",
    "Full Body (Standing Lookbook)",
    "Tabletop / Luxury (Perfume Bottle / Small Product)",
    "Industrial Design (Chair / Medium Product)",
    "Automotive Cove (Car / Large Vehicle)",
    "Volumetric Cyc (Crowds / Dynamic)",
    "Auto (Detect from Prompt)",
]


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
    Luminance-governed extended Reinhard tone reproduction.
    Guarantees zero hue-shift clipping, preserves chromaticity ratios,
    and maintains non-zero highlight derivative (C'(x) > 0).
    """
    lum_in = rgb_to_luminance(x)
    w2 = white_point * white_point
    lum_out = (lum_in * (1.0 + (lum_in / w2))) / (1.0 + lum_in)
    scale = lum_out / torch.clamp(lum_in, min=1e-6)
    return x * scale


def composite_photometric_relight(
    base_image_srgb: torch.Tensor,
    diffuse_irradiance_linear: torch.Tensor,
    specular_irradiance_linear: torch.Tensor,
    ambient_floor: float = 0.05,
    chroma_bleed: float = 0.35,
    highlight_white_point: float = 3.5,
) -> torch.Tensor:
    """
    Production-grade VFX photometric lighting transfer.
    Linearizes base image, applies dynamic ambient floor and luminance-governed Kelvin transfer,
    adds spectrally neutral dielectric specular reflections (catchlights),
    and applies filmic tone compression before returning display sRGB.
    """
    device = base_image_srgb.device

    # 1. Linearize base image into scene-referred physical float space
    base_linear = srgb_to_linear(base_image_srgb)

    # 2. Dynamic Ambient Modulation (Single-source of truth)
    amb = max(float(ambient_floor), 0.02)
    diff_gain = amb + (1.0 - amb) * diffuse_irradiance_linear.to(device)

    # 3. Luminance-Preserving Kelvin Transfer
    diff_lum = rgb_to_luminance(diff_gain)
    norm_chroma = diff_gain / torch.clamp(diff_lum, min=1e-5)
    effective_gain = (1.0 - chroma_bleed) * diff_lum + chroma_bleed * norm_chroma * diff_lum

    # 4. Modulate Diffuse Base
    lit_diffuse = base_linear * effective_gain

    # 5. Add Dielectric Specular (neutral reflection, never multiplied by skin albedo)
    lit_hdr = lit_diffuse + specular_irradiance_linear.to(device)

    # 6. Smooth Highlight Tone Compression (Filmic shoulder, zero yellow shift)
    lit_toned = filmic_reinhard_tonemap(lit_hdr, white_point=highlight_white_point)

    # 7. Encode back to display sRGB
    return linear_to_srgb(lit_toned)


class StudioLightRig:
    """
    Three-Point Studio Lighting Console.
    Controls Key, Fill, and Rim lights using photographic angles, Kelvin temperatures,
    and optical modifiers. Generates both spatial relighting maps and semantic conditioning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = list_presets()
        modifier_list = list(MODIFIER_PRESETS.keys())

        return {
            "required": {
                "preset": (preset_list, {
                    "default": "Custom",
                }),
                "framing": (FRAMING_OPTIONS, {
                    "default": "Portrait (Classical Bust)",
                }),
                "ambient_intensity": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 0.40,
                    "step": 0.01,
                }),
                "invert_normals_y": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Invert Y (DirectX)",
                    "label_off": "Standard (OpenGL)",
                }),

                # --- KEY LIGHT ---
                "key_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON",
                    "label_off": "OFF",
                }),
                "key_azimuth": ("FLOAT", {
                    "default": 45.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                }),
                "key_elevation": ("FLOAT", {
                    "default": 35.0,
                    "min": -30.0,
                    "max": 90.0,
                    "step": 1.0,
                }),
                "key_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                }),
                "key_kelvin": ("INT", {
                    "default": 5500,
                    "min": 2000,
                    "max": 10000,
                    "step": 100,
                }),
                "key_modifier": (modifier_list, {
                    "default": "Octabox (Soft)",
                }),

                # --- FILL LIGHT ---
                "fill_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON",
                    "label_off": "OFF",
                }),
                "fill_azimuth": ("FLOAT", {
                    "default": -30.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                }),
                "fill_elevation": ("FLOAT", {
                    "default": 15.0,
                    "min": -30.0,
                    "max": 90.0,
                    "step": 1.0,
                }),
                "fill_intensity": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "fill_kelvin": ("INT", {
                    "default": 4800,
                    "min": 2000,
                    "max": 10000,
                    "step": 100,
                }),
                "fill_modifier": (modifier_list, {
                    "default": "Octabox (Soft)",
                }),

                # --- RIM / KICKER LIGHT ---
                "rim_enabled": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON",
                    "label_off": "OFF",
                }),
                "rim_azimuth": ("FLOAT", {
                    "default": 160.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                }),
                "rim_elevation": ("FLOAT", {
                    "default": 45.0,
                    "min": -30.0,
                    "max": 90.0,
                    "step": 1.0,
                }),
                "rim_intensity": ("FLOAT", {
                    "default": 1.10,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                }),
                "rim_kelvin": ("INT", {
                    "default": 6500,
                    "min": 2000,
                    "max": 10000,
                    "step": 100,
                }),
                "rim_modifier": (modifier_list, {
                    "default": "Stripbox",
                }),

                # --- SCHEDULING & CONDITIONING (PromptBlock Integration) ---
                "start_step": ("FLOAT", {
                    "default": 0.00,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "end_step": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "conditioning_strength": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "weight_type": (["A_NORM", "ADDITIVE"], {
                    "default": "A_NORM",
                }),
            },
            "optional": {
                "normals_optional": ("IMAGE",),
                "depth_optional": ("IMAGE",),
                "image_optional": ("IMAGE",),
                "clip_optional": ("CLIP",),
                "conditioning_in": ("CONDITIONING",),
                "prompt_text_optional": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CONDITIONING", "STRING", "IMAGE")
    RETURN_NAMES = ("relighting_map", "probe_preview", "conditioning", "lighting_prompt", "lit_composite")
    FUNCTION = "execute_rig"
    CATEGORY = "lighting"
    DESCRIPTION = (
        "ComfyUI-StudioLight: Professional Photographic 3-Point Studio Rig.\n"
        "• relighting_map: 32-bit deferred physical lighting pass.\n"
        "• probe_preview: VFX 18% Grey Ball + Mirror Chrome Ball + 2D Studio Radar.\n"
        "• conditioning: Step-scheduled prompt latents compatible with PromptBlock (0.40-1.00).\n"
        "• lighting_prompt: Concise, high-signal photographic tokens for Flux/SDXL.\n"
        "• lit_composite: Photometric composite of image_optional * relighting_map for refinement."
    )

    def execute_rig(
        self,
        preset: str,
        framing: str = "Portrait (Classical Bust)",
        ambient_intensity: float = 0.05,
        invert_normals_y: bool = False,
        key_enabled: bool = True,
        key_azimuth: float = 45.0,
        key_elevation: float = 35.0,
        key_intensity: float = 1.0,
        key_kelvin: int = 5500,
        key_modifier: str = "Octabox (Soft)",
        fill_enabled: bool = True,
        fill_azimuth: float = -30.0,
        fill_elevation: float = 15.0,
        fill_intensity: float = 0.35,
        fill_kelvin: int = 4800,
        fill_modifier: str = "Octabox (Soft)",
        rim_enabled: bool = True,
        rim_azimuth: float = 160.0,
        rim_elevation: float = 45.0,
        rim_intensity: float = 1.1,
        rim_kelvin: int = 6500,
        rim_modifier: str = "Stripbox",
        start_step: float = 0.0,
        end_step: float = 1.0,
        conditioning_strength: float = 1.0,
        weight_type: str = "A_NORM",
        normals_optional: Optional[torch.Tensor] = None,
        depth_optional: Optional[torch.Tensor] = None,
        image_optional: Optional[torch.Tensor] = None,
        clip_optional: Optional[Any] = None,
        conditioning_in: Optional[List] = None,
        prompt_text_optional: Optional[str] = None,
        prompt: Optional[Dict] = None,
        unique_id: Optional[str] = None,
        extra_pnginfo: Optional[Dict] = None,
        **kwargs,
    ):
        # Direct parameter pass-through: Any adjustments made on the node face take immediate effect
        key_params = {
            "enabled": key_enabled,
            "azimuth": key_azimuth,
            "elevation": key_elevation,
            "intensity": key_intensity,
            "kelvin": key_kelvin,
            "modifier": key_modifier,
        }
        fill_params = {
            "enabled": fill_enabled,
            "azimuth": fill_azimuth,
            "elevation": fill_elevation,
            "intensity": fill_intensity,
            "kelvin": fill_kelvin,
            "modifier": fill_modifier,
        }
        rim_params = {
            "enabled": rim_enabled,
            "azimuth": rim_azimuth,
            "elevation": rim_elevation,
            "intensity": rim_intensity,
            "kelvin": rim_kelvin,
            "modifier": rim_modifier,
        }

        # 2. Extract detected prompt text for "Auto (Detect from Prompt)"
        detected_prompt = ""
        if prompt_text_optional:
            detected_prompt = str(prompt_text_optional).strip()
        elif prompt and unique_id:
            try:
                # Check directly connected node via conditioning_in
                node_info = prompt.get(str(unique_id), {})
                inputs = node_info.get("inputs", {})
                cond_link = inputs.get("conditioning_in")
                if isinstance(cond_link, list) and len(cond_link) > 0:
                    up_id = str(cond_link[0])
                    if up_id in prompt:
                        up_inputs = prompt[up_id].get("inputs", {})
                        for k in ["text", "prompt", "text_positive", "positive"]:
                            if k in up_inputs and isinstance(up_inputs[k], str):
                                detected_prompt = up_inputs[k]
                                break
                # Fallback: scan all prompt blocks in the graph for subject keywords
                if not detected_prompt:
                    for nid, nval in prompt.items():
                        ninputs = nval.get("inputs", {})
                        for k in ["text", "prompt"]:
                            if k in ninputs and isinstance(ninputs[k], str) and len(ninputs[k]) > 10:
                                detected_prompt = ninputs[k]
                                break
                        if detected_prompt:
                            break
            except Exception as e:
                print(f"[StudioLight] Note: Prompt auto-detection skipped: {e}")

        # 3. Render Physical Relighting Map and Dual-Sphere VFX Probe
        relighting_map, probe_preview, diff_pass, spec_pass = render_studio_lighting(
            normals=normals_optional,
            depth=depth_optional,
            key_params=key_params,
            fill_params=fill_params,
            rim_params=rim_params,
            ambient_intensity=ambient_intensity,
            invert_normals_y=invert_normals_y,
            preview_resolution=512,
            framing=framing,
            detected_prompt=detected_prompt,
            return_passes=True,
        )

        # 3. Synthesize High-Signal Photographic Prompt Tokens
        target_engine = kwargs.get("target_engine", "Flux.1 / Flux.2")
        if detected_prompt and target_engine == "Flux.1 / Flux.2":
            p_low = detected_prompt.lower()
            if any(w in p_low for w in ["video", "cinematic shot", "camera pan", "motion", "wan", "minimax"]):
                target_engine = "Video (Wan / MiniMax)"
            elif any(w in p_low for w in ["sdxl", "score_9", "source_pony"]):
                target_engine = "SDXL"

        lighting_prompt = synthesize_high_signal_prompt(
            key_params, fill_params, rim_params, framing=framing, target_engine=target_engine
        )

        # 4. Encode Step-Scheduled Conditioning (PromptBlock integration)
        conditioning_out = encode_scheduled_conditioning(
            clip=clip_optional,
            text=lighting_prompt,
            start_step=start_step,
            end_step=end_step,
            strength=conditioning_strength,
            weight_type=weight_type,
            conditioning_in=conditioning_in,
        )

        # 5. Photometric Composite (if image_optional is provided)
        if image_optional is not None:
            comp_img = image_optional.clone()
            if comp_img.shape[-1] != 3 and comp_img.shape[1] == 3:
                comp_img = comp_img.permute(0, 2, 3, 1)

            # Match spatial dimensions if needed
            if comp_img.shape[1:3] != diff_pass.shape[1:3]:
                d_perm = diff_pass.permute(0, 3, 1, 2)
                s_perm = spec_pass.permute(0, 3, 1, 2)
                d_resized = torch.nn.functional.interpolate(
                    d_perm, size=comp_img.shape[1:3], mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                s_resized = torch.nn.functional.interpolate(
                    s_perm, size=comp_img.shape[1:3], mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
            else:
                d_resized = diff_pass
                s_resized = spec_pass

            # CRITICAL SAFEGUARD: If normals_optional was NOT provided, diff_pass
            # contains the built-in static mannequin bust normals and its alpha stencil mask!
            # Stamping a static mannequin cutout onto an arbitrary photo creates a ghost artifact.
            # In this case, synthesize a smooth full-frame directional lighting gradient instead.
            if normals_optional is None:
                h, w = comp_img.shape[1], comp_img.shape[2]
                dev = comp_img.device
                y_g, x_g = torch.meshgrid(
                    torch.linspace(-1.0, 1.0, h, device=dev),
                    torch.linspace(-1.0, 1.0, w, device=dev),
                    indexing="ij"
                )
                nx = x_g * 0.55
                ny = -y_g * 0.35
                nz = torch.sqrt(torch.clamp(1.0 - nx**2 - ny**2, min=0.15))
                smooth_norm = torch.stack([nx, ny, nz], dim=-1).unsqueeze(0)
                smooth_norm = smooth_norm / torch.norm(smooth_norm, dim=-1, keepdim=True)

                d_fallback = torch.zeros_like(smooth_norm)
                s_fallback = torch.zeros_like(smooth_norm)
                for p_dict in [key_params, fill_params, rim_params]:
                    if p_dict.get("enabled", True):
                        p_vec = azimuth_elevation_to_vector(p_dict["azimuth"], p_dict["elevation"])
                        p_col = kelvin_to_rgb(p_dict["kelvin"])
                        d_sub, s_sub = compute_light_shading(
                            smooth_norm, p_vec, p_col, p_dict["intensity"], p_dict["modifier"], depth=None, return_separate=True
                        )
                        d_fallback = d_fallback + d_sub
                        s_fallback = s_fallback + s_sub
                d_resized = d_fallback
                s_resized = s_fallback

            lit_composite = composite_photometric_relight(
                base_image_srgb=comp_img,
                diffuse_irradiance_linear=d_resized,
                specular_irradiance_linear=s_resized,
                ambient_floor=ambient_intensity,
                chroma_bleed=0.35,
                highlight_white_point=3.5,
            )
        else:
            lit_composite = relighting_map

        return (relighting_map, probe_preview, conditioning_out, lighting_prompt, lit_composite)


NODE_CLASS_MAPPINGS = {
    "StudioLightRig": StudioLightRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightRig": "Studio Light Rig (3-Point Photographic Console)",
}
