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
from .preset_manager import list_presets, get_all_presets_flattened
from .shader_engine import render_studio_lighting, AIM_TARGET_OPTIONS
from .prompt_synthesizer import synthesize_high_signal_prompt


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
                    "default": 1.35,
                    "min": 1.0,
                    "max": 4.0,
                    "step": 0.05,
                }),
                "ao_anchoring": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "normals_optional": ("IMAGE",),
                "depth_optional": ("IMAGE",),
                "invert_normals_x": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Invert X (DSINE / Vision)",
                    "label_off": "Standard (OpenGL / Blender)",
                }),
                "target_engine": (["Flux.1", "SDXL"], {
                    "default": "Flux.1",
                }),
                # NOTE: widget-type optionals serialize in order AFTER the 26 required widgets (26 + 5 = 31 total).
                # Only ever APPEND new widgets here, never insert, or old workflows shift.
                "invert_depth": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Inverted (near = black)",
                    "label_off": "Standard (near = white)",
                }),
                "aim_target": (AIM_TARGET_OPTIONS, {
                    "default": "Upper",
                }),
                "depth_relief_scale": ("FLOAT", {
                    "default": 0.30,
                    "min": 0.10,
                    "max": 1.00,
                    "step": 0.05,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "IMAGE", "STUDIOLIGHT_TELEMETRY")
    RETURN_NAMES = ("relighting_map", "probe_preview", "lighting_prompt", "specular_pass", "telemetry")
    FUNCTION = "execute_rig_with_telemetry"
    CATEGORY = "StudioLight/Rig"
    DESCRIPTION = (
        "ComfyUI-StudioLight: Professional Photographic 3-Point Studio Rig.\n"
        "• relighting_map: 32-bit deferred physical lighting pass (Lit proxy plate for Flux).\n"
        "• probe_preview: VFX 18% Grey Ball + Mirror Chrome Ball + 2D Studio Radar.\n"
        "• lighting_prompt: Concise, high-signal photographic tokens for Flux/SDXL.\n"
        "• specular_pass: isolated Blinn-Phong specular lobes (feed StudioPhotometricComposite.specular_map)."
    )

    def execute_rig_with_telemetry(self, **kwargs):
        """ComfyUI entrypoint; execute_rig keeps its four-output Python API."""
        result = self.execute_rig(**kwargs)
        return (*result, getattr(result[0], "_studiolight_telemetry", {}))

    def execute_rig(
        self,
        preset: str,
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
        normal_boost: float = 1.35,
        ao_anchoring: float = 0.65,
        normals_optional: Optional[torch.Tensor] = None,
        depth_optional: Optional[torch.Tensor] = None,
        invert_normals_x: bool = True,
        target_engine: str = "Flux.1",
        invert_depth: bool = False,
        aim_target: str = "Upper",
        depth_relief_scale: float = 0.30,
        **kwargs,
    ):
        # 1. Server-side preset resolution (ensures API, headless, and Python invocations honor `preset`)
        if preset and preset != "Custom":
            p_flat = get_all_presets_flattened().get(preset)
            if p_flat:
                ambient_intensity = float(p_flat.get("ambient_intensity", ambient_intensity))
                key_enabled = bool(p_flat.get("key_enabled", key_enabled))
                key_azimuth = float(p_flat.get("key_azimuth", key_azimuth))
                key_elevation = float(p_flat.get("key_elevation", key_elevation))
                key_distance = float(p_flat.get("key_distance", key_distance))
                key_intensity = float(p_flat.get("key_intensity", key_intensity))
                key_kelvin = int(p_flat.get("key_kelvin", key_kelvin))
                key_modifier = str(p_flat.get("key_modifier", key_modifier))
                fill_enabled = bool(p_flat.get("fill_enabled", fill_enabled))
                fill_azimuth = float(p_flat.get("fill_azimuth", fill_azimuth))
                fill_elevation = float(p_flat.get("fill_elevation", fill_elevation))
                fill_distance = float(p_flat.get("fill_distance", fill_distance))
                fill_intensity = float(p_flat.get("fill_intensity", fill_intensity))
                fill_kelvin = int(p_flat.get("fill_kelvin", fill_kelvin))
                fill_modifier = str(p_flat.get("fill_modifier", fill_modifier))
                rim_enabled = bool(p_flat.get("rim_enabled", rim_enabled))
                rim_azimuth = float(p_flat.get("rim_azimuth", rim_azimuth))
                rim_elevation = float(p_flat.get("rim_elevation", rim_elevation))
                rim_distance = float(p_flat.get("rim_distance", rim_distance))
                rim_intensity = float(p_flat.get("rim_intensity", rim_intensity))
                rim_kelvin = int(p_flat.get("rim_kelvin", rim_kelvin))
                rim_modifier = str(p_flat.get("rim_modifier", rim_modifier))
                normal_boost = float(p_flat.get("normal_boost", normal_boost))
                ao_anchoring = float(p_flat.get("ao_anchoring", ao_anchoring))

        active_normals = normals_optional
        active_depth = depth_optional
        effective_invert_x = invert_normals_x
        target_pos = kwargs.get("target_pos", None)
        vertical_fov_deg = float(kwargs.get("vertical_fov_deg", 40.0))

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

        # 2. Render Relighting Map and Dual-Sphere VFX Probe
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
            return_passes=True,
            invert_normals_x=effective_invert_x,
            invert_depth=bool(invert_depth),
            target_pos=target_pos,
            vertical_fov_deg=vertical_fov_deg,
            aim_target=aim_target,
            depth_relief_scale=float(depth_relief_scale),
        )

        # 3. Synthesize High-Signal Photographic Prompt Tokens
        lighting_prompt = synthesize_high_signal_prompt(
            key_params, fill_params, rim_params, target_engine=target_engine
        )

        specular_pass = torch.clamp(spec_pass.float(), min=0.0)
        if hasattr(relighting_map, "_studiolight_telemetry") and isinstance(relighting_map._studiolight_telemetry, dict):
            relighting_map._studiolight_telemetry["lights"] = [key_params, fill_params, rim_params]
            relighting_map._studiolight_telemetry["camera_fov_deg"] = float(vertical_fov_deg)
        return (relighting_map, probe_preview, lighting_prompt, specular_pass)


NODE_CLASS_MAPPINGS = {
    "StudioLightRig": StudioLightRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightRig": "Studio Light Rig (3-Point Photographic Console)",
}
