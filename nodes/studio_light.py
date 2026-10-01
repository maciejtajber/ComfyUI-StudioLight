"""
ComfyUI-StudioLight — Professional Photographic Studio Lighting Controller
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Flagship 3-point photographic lighting console node for ComfyUI.
Integrates physics-based deferred normal relighting, high-signal photographic
prompt synthesis, and visual VFX probe previews.
"""

from typing import Dict, Any, Optional
import torch
from .color_utils import MODIFIER_PRESETS
from .preset_manager import list_presets
from .shader_engine import render_studio_lighting
from .prompt_synthesizer import synthesize_high_signal_prompt


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
    and optical modifiers. Generates physical lit proxy plate for Flux and visual VFX probes.
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
                "key_distance": ("FLOAT", {
                    "default": 1.8,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
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
                "fill_distance": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
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
                "rim_distance": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
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

                # --- 3D VOLUME & SHADOW SCULPTING (Intervention Engine) ---
                "normal_boost": ("FLOAT", {
                    "default": 2.50,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.1,
                }),
                "ao_anchoring": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "scene_3d": ("SCENE_3D",),
                "normals_optional": ("IMAGE",),
                "depth_optional": ("IMAGE",),
                "invert_normals_x": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Invert X (DSINE / Vision)",
                    "label_off": "Standard (OpenGL / Blender)",
                }),
                "target_engine": (["Flux.1 / Flux.2", "Video (Wan / MiniMax)", "SDXL"], {
                    "default": "Flux.1 / Flux.2",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("relighting_map", "probe_preview", "lighting_prompt")
    FUNCTION = "execute_rig"
    CATEGORY = "lighting"
    DESCRIPTION = (
        "ComfyUI-StudioLight: Professional Photographic 3-Point Studio Rig.\n"
        "• relighting_map: 32-bit deferred physical lighting pass (Lit proxy plate for Flux).\n"
        "• probe_preview: VFX 18% Grey Ball + Mirror Chrome Ball + 2D Studio Radar.\n"
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
        key_distance: float = 1.8,
        key_intensity: float = 1.0,
        key_kelvin: int = 5500,
        key_modifier: str = "Octabox (Soft)",
        fill_enabled: bool = True,
        fill_azimuth: float = -30.0,
        fill_elevation: float = 15.0,
        fill_distance: float = 2.5,
        fill_intensity: float = 0.35,
        fill_kelvin: int = 4800,
        fill_modifier: str = "Octabox (Soft)",
        rim_enabled: bool = True,
        rim_azimuth: float = 160.0,
        rim_elevation: float = 45.0,
        rim_distance: float = 1.5,
        rim_intensity: float = 1.1,
        rim_kelvin: int = 6500,
        rim_modifier: str = "Stripbox",
        normal_boost: float = 2.50,
        ao_anchoring: float = 0.65,
        scene_3d: Optional[Dict[str, Any]] = None,
        normals_optional: Optional[torch.Tensor] = None,
        depth_optional: Optional[torch.Tensor] = None,
        invert_normals_x: bool = True,
        target_engine: str = "Flux.1 / Flux.2",
        prompt: Optional[Dict] = None,
        unique_id: Optional[str] = None,
        extra_pnginfo: Optional[Dict] = None,
        **kwargs,
    ):
        # 1. Unpack scene_3d if provided
        active_normals = normals_optional
        active_depth = depth_optional
        effective_invert_x = invert_normals_x
        if scene_3d is not None and isinstance(scene_3d, dict):
            if active_normals is None:
                active_normals = scene_3d.get("normals")
            if active_depth is None:
                active_depth = scene_3d.get("depth")
            if "framing" in scene_3d.get("metadata", {}) and framing == "Portrait (Classical Bust)":
                framing = scene_3d["metadata"]["framing"]
            if scene_3d.get("metadata", {}).get("is_stage_asset", False) or scene_3d.get("metadata", {}).get("is_built_in", False):
                effective_invert_x = False

        key_params = {
            "enabled": key_enabled,
            "azimuth": key_azimuth,
            "elevation": key_elevation,
            "intensity": key_intensity,
            "kelvin": key_kelvin,
            "modifier": key_modifier,
            "distance": float(key_distance),
        }
        fill_params = {
            "enabled": fill_enabled,
            "azimuth": fill_azimuth,
            "elevation": fill_elevation,
            "intensity": fill_intensity,
            "kelvin": fill_kelvin,
            "modifier": fill_modifier,
            "distance": float(fill_distance),
        }
        rim_params = {
            "enabled": rim_enabled,
            "azimuth": rim_azimuth,
            "elevation": rim_elevation,
            "intensity": rim_intensity,
            "kelvin": rim_kelvin,
            "modifier": rim_modifier,
            "distance": float(rim_distance),
        }

        # 2. Render Physical Relighting Map (Lit proxy plate) and Dual-Sphere VFX Probe
        relighting_map, probe_preview, diff_pass, spec_pass = render_studio_lighting(
            normals=active_normals,
            depth=active_depth,
            key_params=key_params,
            fill_params=fill_params,
            rim_params=rim_params,
            ambient_intensity=ambient_intensity,
            invert_normals_y=invert_normals_y,
            normal_boost=normal_boost,
            ao_anchoring=ao_anchoring,
            preview_resolution=512,
            framing=framing,
            detected_prompt="",
            return_passes=True,
            invert_normals_x=effective_invert_x,
        )

        # 3. Synthesize High-Signal Photographic Prompt Tokens
        lighting_prompt = synthesize_high_signal_prompt(
            key_params, fill_params, rim_params, framing=framing, target_engine=target_engine
        )

        return (relighting_map, probe_preview, lighting_prompt)


NODE_CLASS_MAPPINGS = {
    "StudioLightRig": StudioLightRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightRig": "Studio Light Rig (3-Point Photographic Console)",
}
