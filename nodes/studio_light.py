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
from .shader_engine import render_studio_lighting
from .prompt_synthesizer import synthesize_high_signal_prompt, encode_scheduled_conditioning


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
                    "default": 0.40,
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
                "clip_optional": ("CLIP",),
                "conditioning_in": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("relighting_map", "probe_preview", "conditioning", "lighting_prompt")
    FUNCTION = "execute_rig"
    CATEGORY = "lighting"
    DESCRIPTION = (
        "ComfyUI-StudioLight: Professional Photographic 3-Point Studio Rig.\n"
        "• relighting_map: 32-bit deferred physical lighting pass for ControlNet / IC-Light.\n"
        "• probe_preview: VFX 18% Grey Ball + Mirror Chrome Ball + 2D Studio Radar.\n"
        "• conditioning: Step-scheduled prompt latents compatible with PromptBlock (0.40-1.00).\n"
        "• lighting_prompt: Concise, high-signal photographic tokens for Flux/SDXL."
    )

    def execute_rig(
        self,
        preset: str,
        ambient_intensity: float,
        invert_normals_y: bool,
        key_enabled: bool,
        key_azimuth: float,
        key_elevation: float,
        key_intensity: float,
        key_kelvin: int,
        key_modifier: str,
        fill_enabled: bool,
        fill_azimuth: float,
        fill_elevation: float,
        fill_intensity: float,
        fill_kelvin: int,
        fill_modifier: str,
        rim_enabled: bool,
        rim_azimuth: float,
        rim_elevation: float,
        rim_intensity: float,
        rim_kelvin: int,
        rim_modifier: str,
        start_step: float,
        end_step: float,
        conditioning_strength: float,
        weight_type: str,
        normals_optional: Optional[torch.Tensor] = None,
        depth_optional: Optional[torch.Tensor] = None,
        clip_optional: Optional[Any] = None,
        conditioning_in: Optional[List] = None,
    ):
        # 1. Check if a non-custom preset was selected
        if preset != "Custom":
            p_data = load_preset_data(preset)
            if p_data:
                ambient_intensity = p_data.get("ambient_intensity", ambient_intensity)
                k = p_data.get("key", {})
                key_enabled = k.get("enabled", key_enabled)
                key_azimuth = k.get("azimuth", key_azimuth)
                key_elevation = k.get("elevation", key_elevation)
                key_intensity = k.get("intensity", key_intensity)
                key_kelvin = k.get("kelvin", key_kelvin)
                key_modifier = k.get("modifier", key_modifier)

                f = p_data.get("fill", {})
                fill_enabled = f.get("enabled", fill_enabled)
                fill_azimuth = f.get("azimuth", fill_azimuth)
                fill_elevation = f.get("elevation", fill_elevation)
                fill_intensity = f.get("intensity", fill_intensity)
                fill_kelvin = f.get("kelvin", fill_kelvin)
                fill_modifier = f.get("modifier", fill_modifier)

                r = p_data.get("rim", {})
                rim_enabled = r.get("enabled", rim_enabled)
                rim_azimuth = r.get("azimuth", rim_azimuth)
                rim_elevation = r.get("elevation", rim_elevation)
                rim_intensity = r.get("intensity", rim_intensity)
                rim_kelvin = r.get("kelvin", rim_kelvin)
                rim_modifier = r.get("modifier", rim_modifier)

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

        # 2. Render Physical Relighting Map and Dual-Sphere VFX Probe
        relighting_map, probe_preview = render_studio_lighting(
            normals=normals_optional,
            depth=depth_optional,
            key_params=key_params,
            fill_params=fill_params,
            rim_params=rim_params,
            ambient_intensity=ambient_intensity,
            invert_normals_y=invert_normals_y,
            preview_resolution=512,
        )

        # 3. Synthesize High-Signal Photographic Prompt Tokens
        lighting_prompt = synthesize_high_signal_prompt(key_params, fill_params, rim_params)

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

        return (relighting_map, probe_preview, conditioning_out, lighting_prompt)


NODE_CLASS_MAPPINGS = {
    "StudioLightRig": StudioLightRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightRig": "Studio Light Rig (3-Point Photographic Console)",
}
