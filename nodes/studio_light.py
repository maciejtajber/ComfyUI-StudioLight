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
        relighting_map, probe_preview = render_studio_lighting(
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
        )

        # 3. Synthesize High-Signal Photographic Prompt Tokens
        lighting_prompt = synthesize_high_signal_prompt(key_params, fill_params, rim_params, framing=framing)

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
