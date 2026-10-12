"""
ComfyUI-StudioLight — High-Signal Photographic Prompt Synthesizer
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Translates studio lighting physics into concise, high-signal photographic tokens
optimized for Flux.1 and SDXL attention without prompt pollution.
"""

from typing import Dict, Any, Optional, Tuple, List
from .color_utils import kelvin_to_mood, MODIFIER_PRESETS


def get_azimuth_direction_token(
    azimuth_deg: float,
    elevation_deg: float = 35.0,
    target_engine: str = "Flux.1",
) -> str:
    """
    Translates numeric azimuth & elevation into frame-relative directional language.
    Eliminates photographic jargon confusion (e.g. 'camera-right' vs 'subject-right').
    Optimized for Flux.1/2 (T5-XXL), Wan 2.1 / MiniMax (Video blocks), and SDXL (CLIP).
    """
    az = float(azimuth_deg)
    elev = float(elevation_deg)

    if "SDXL" in target_engine:
        if abs(az) <= 18:
            return "frontal on-axis"
        elif 18 < az <= 65:
            return "right side 45 degree"
        elif -65 <= az < -18:
            return "left side 45 degree"
        elif 65 < az <= 115:
            return "right profile 90 degree"
        elif -115 <= az < -65:
            return "left profile 90 degree"
        elif 115 < az <= 165:
            return "rear right rim kicker"
        elif -165 <= az < -115:
            return "rear left rim kicker"
        else:
            return "direct rear backlight"

    # Default: Flux.1 / Flux.2 & Video narrative cause-and-effect
    if elev > 50:
        elev_str = "steep high-angle overhead"
    elif elev > 15:
        elev_str = f"angled from {int(round(elev))} degrees above"
    elif elev >= -10:
        elev_str = "at eye level"
    else:
        elev_str = f"low-angle uplight from {int(round(abs(elev)))} degrees below"

    if abs(az) <= 18:
        return f"directly in front on-axis {elev_str}, illuminating the front of the face with minimal side shadows"
    elif 18 < az <= 65:
        return f"positioned to the right side of the frame {elev_str}, illuminating the right cheek and casting soft directional shadows across the left side of the face"
    elif -65 <= az < -18:
        return f"positioned to the left side of the frame {elev_str}, illuminating the left cheek and casting soft directional shadows across the right side of the face"
    elif 65 < az <= 115:
        return f"positioned at the extreme right 90-degree profile edge of the frame {elev_str}, creating high-contrast chiaroscuro split lighting with deep shadows on the left"
    elif -115 <= az < -65:
        return f"positioned at the extreme left 90-degree profile edge of the frame {elev_str}, creating high-contrast chiaroscuro split lighting with deep shadows on the right"
    elif 115 < az <= 165:
        return f"positioned behind the subject toward the rear right {elev_str}, casting a bright rim kicker edge along the right silhouette"
    elif -165 <= az < -115:
        return f"positioned behind the subject toward the rear left {elev_str}, casting a bright rim kicker edge along the left silhouette"
    else:
        return f"positioned directly behind the subject {elev_str}, casting a strong backlight halo silhouette"


def synthesize_high_signal_prompt(
    key_params: Dict[str, Any],
    fill_params: Dict[str, Any],
    rim_params: Dict[str, Any],
    target_engine: str = "Flux.1",
) -> str:
    """
    Synthesizes a curated high-signal photographic prompt tailored to the target neural architecture.
    - Flux.1: Descriptive cause-and-effect narrative grounded in physical optical anchors.
    - SDXL: Weighted, comma-delimited photographic tags.
    """
    k_enabled = key_params.get("enabled", True)
    f_enabled = fill_params.get("enabled", True) and fill_params.get("intensity", 0.0) > 0.05
    r_enabled = rim_params.get("enabled", True) and rim_params.get("intensity", 0.0) > 0.1

    k_az = key_params.get("azimuth", 45.0)
    k_el = key_params.get("elevation", 35.0)
    k_dir = get_azimuth_direction_token(k_az, k_el, target_engine)
    k_mood = kelvin_to_mood(key_params.get("kelvin", 5500))
    k_mod_raw = key_params.get("modifier", "Octabox (Soft)")
    k_mod_name = k_mod_raw.split(" (")[0] if " (" in k_mod_raw else k_mod_raw
    k_kelvin = int(key_params.get("kelvin", 5500))

    f_az = fill_params.get("azimuth", -30.0)
    f_el = fill_params.get("elevation", 15.0)
    f_dir = get_azimuth_direction_token(f_az, f_el, target_engine)
    f_mood = kelvin_to_mood(fill_params.get("kelvin", 4800))

    r_az = rim_params.get("azimuth", 160.0)
    r_el = rim_params.get("elevation", 45.0)
    r_dir = get_azimuth_direction_token(r_az, r_el, target_engine)
    r_mood = kelvin_to_mood(rim_params.get("kelvin", 6500))

    # --- VIDEO ENGINE (WAN 2.1 / MINIMAX / KLING) ---
    if "Video" in target_engine:
        video_blocks = ["[Cinematic studio lighting]", "[Stationary three-point light setup]"]
        if k_enabled:
            video_blocks.append(f"[Fixed key light from a studio {k_mod_name} {k_dir}, steady {k_kelvin}K {k_mood}]")
        if f_enabled:
            fill_side = "left" if f_az < -10 else ("right" if f_az > 10 else "front")
            video_blocks.append(f"[Subtle {f_mood} ambient fill light from the {fill_side} side]")
        if r_enabled:
            video_blocks.append(f"[Steady {r_mood} rim kicker from behind for subject separation]")
        video_blocks.append("[Consistent color temperature, temporally stable lighting, no flickering shadows].")
        return " + ".join(video_blocks)

    # --- SDXL ENGINE (COMMA-DELIMITED TAGS) ---
    if "SDXL" in target_engine:
        tags = ["studio portrait, editorial fashion photography"]
        if k_enabled:
            tags.append(f"key light from {k_dir}, studio {k_mod_name}, {k_kelvin}K {k_mood}")
        if f_enabled:
            tags.append(f"soft fill light from {f_dir}")
        if r_enabled:
            tags.append(f"rim kicker from {r_dir}, {r_mood}")
        tags.extend(["chiaroscuro", "calibrated studio lighting", "8k photography", "high dynamic range"])
        return ", ".join(tags)

    # --- FLUX.1 / FLUX.2 ENGINE (NARRATIVE CAUSE & EFFECT) ---
    clauses = []
    if k_enabled:
        clauses.append(f"Key light from a studio {k_mod_name} {k_dir}, calibrated {k_kelvin}K {k_mood}")
    if f_enabled:
        if f_az < -10:
            fill_side = "left side of the frame"
        elif f_az > 10:
            fill_side = "right side of the frame"
        else:
            fill_side = "front on-axis"
        clauses.append(f"subtle {f_mood} fill light from the {fill_side} to soften shadow density")
    if r_enabled:
        clauses.append(f"crisp {r_mood} rim kicker from behind creating a thin razor-edge highlight along the hair and shoulders")

    lighting_narrative = ". ".join(clauses) if clauses else "clean calibrated studio ambient illumination"
    return f"Studio fashion portrait. {lighting_narrative}. Calibrated dynamic range, natural skin subsurface scattering, high-end editorial fashion photography."


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
