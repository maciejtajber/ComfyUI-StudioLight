"""
ComfyUI-StudioLight — High-Signal Photographic Prompt Synthesizer
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Translates studio lighting physics into concise, high-signal photographic tokens
optimized for Flux.1 and SDXL attention without prompt pollution.
"""

from typing import Dict, Any, Optional, Tuple, List
import torch
from .color_utils import kelvin_to_mood, MODIFIER_PRESETS


def get_azimuth_direction_token(azimuth_deg: float) -> str:
    """Translates numeric azimuth into concise, high-signal camera-relative direction."""
    az = float(azimuth_deg)
    if abs(az) <= 18:
        return "frontal on-axis"
    elif 18 < az <= 65:
        return "camera-right Rembrandt"
    elif -65 <= az < -18:
        return "camera-left Rembrandt"
    elif 65 < az <= 115:
        return "camera-right 90-degree profile"
    elif -115 <= az < -65:
        return "camera-left 90-degree profile"
    elif 115 < az <= 165:
        return "camera-right rear kicker"
    elif -165 <= az < -115:
        return "camera-left rear kicker"
    else:
        return "direct backlight silhouette"


def synthesize_high_signal_prompt(
    key_params: Dict[str, Any],
    fill_params: Dict[str, Any],
    rim_params: Dict[str, Any],
    framing: str = "Portrait (Classical Bust)",
) -> str:
    """
    Synthesizes a curated 15-25 word high-signal photographic prompt.
    Eliminates verbose prompt pollution so Flux/SDXL attention focuses on key features.
    Adapts context for fashion portraits, tabletop luxury products, industrial design, and cars.
    """
    phrases = []

    # 1. Key Light
    if key_params.get("enabled", True):
        k_dir = get_azimuth_direction_token(key_params["azimuth"])
        k_mood = kelvin_to_mood(key_params["kelvin"])
        k_mod = MODIFIER_PRESETS.get(key_params["modifier"], {}).get("token", "soft studio light")
        phrases.append(f"{k_mood} key light from {k_dir}, {k_mod}")

    # 2. Fill Light
    if fill_params.get("enabled", True) and fill_params.get("intensity", 0.0) > 0.05:
        f_dir = get_azimuth_direction_token(fill_params["azimuth"])
        phrases.append(f"subtle soft fill from {f_dir}")

    # 3. Rim Light
    if rim_params.get("enabled", True) and rim_params.get("intensity", 0.0) > 0.1:
        r_dir = get_azimuth_direction_token(rim_params["azimuth"])
        r_mood = kelvin_to_mood(rim_params["kelvin"])
        phrases.append(f"crisp {r_mood} rim kicker from {r_dir}")

    lighting_body = ", ".join(phrases) if phrases else "clean volumetric studio shadow falloff"

    # Genre-aware photographic finishing anchor
    if "Tabletop" in framing or "Perfume" in framing:
        return f"commercial product photography, luxury tabletop lighting, {lighting_body}, calibrated dynamic range, editorial advertising photography."
    elif "Industrial" in framing or "Chair" in framing:
        return f"industrial design photography, studio furniture showcase, {lighting_body}, calibrated dynamic range, architectural digest photography."
    elif "Automotive" in framing or "Car" in framing:
        return f"commercial automotive photography, car studio lighting, infinity cove reflections, {lighting_body}, calibrated dynamic range, editorial automotive photography."
    else:
        return f"studio portrait, {lighting_body}, calibrated dynamic range, editorial fashion photography."


def encode_scheduled_conditioning(
    clip: Any,
    text: str,
    start_step: float = 0.40,
    end_step: float = 1.00,
    strength: float = 1.0,
    weight_type: str = "A_NORM",
    conditioning_in: Optional[List] = None,
) -> List:
    """
    Encodes synthesized lighting tokens into step-scheduled conditioning
    fully compatible with ComfyUI-PromptBlock.
    """
    if clip is None or not text:
        return conditioning_in if conditioning_in is not None else []

    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")

    # Magnitude normalization (A_NORM) against upstream chain to prevent blowing out highlights
    if weight_type == "A_NORM" and conditioning_in is not None and len(conditioning_in) > 0:
        prev_cond = conditioning_in[-1][0]
        ref_norm = prev_cond.norm(dim=-1, keepdim=True).mean() + 1e-8
        cond_norm = cond.norm(dim=-1, keepdim=True).mean() + 1e-8
        cond = cond * (ref_norm / cond_norm)

    cond_scaled = cond * float(strength)

    meta = dict(output)
    if start_step > 0.0:
        meta["start_percent"] = float(start_step)
    if end_step < 1.0:
        meta["end_percent"] = float(end_step)
    meta["strength"] = float(strength)

    this_block = [cond_scaled, meta]

    if conditioning_in is None:
        return [this_block]
    return list(conditioning_in) + [this_block]
