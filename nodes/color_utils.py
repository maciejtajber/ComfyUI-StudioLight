"""
ComfyUI-StudioLight — Color Science & Photometric Modifiers
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Implements blackbody Kelvin radiation (1000K-12000K) to linear/sRGB conversions
and optical modifier profiles for studio photography.
"""

import math
from typing import Tuple, Dict, Any


def srgb_channel_to_linear(c: float) -> float:
    """Converts a standard sRGB channel [0, 1] to linear Rec.709 photometric space."""
    return c / 12.92 if c <= 0.04045 else math.pow((c + 0.055) / 1.055, 2.4)


def kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """
    Converts a color temperature in Kelvin (1000K - 12000K) to linear normalized RGB (0.0 - 1.0).
    Uses Tanner Helland Planckian locus approximation converted to true linear Rec.709 radiance.
    """
    temp = max(1000.0, min(12000.0, float(kelvin))) / 100.0

    # Red component
    if temp <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * math.pow(temp - 60.0, -0.1332047592)
        r = max(0.0, min(255.0, r))

    # Green component
    if temp <= 66.0:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        g = max(0.0, min(255.0, g))
    else:
        g = 288.1221695283 * math.pow(temp - 60.0, -0.0755148492)
        g = max(0.0, min(255.0, g))

    # Blue component
    if temp >= 66.0:
        b = 255.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
        b = max(0.0, min(255.0, b))

    sr, sg, sb = r / 255.0, g / 255.0, b / 255.0
    return (
        srgb_channel_to_linear(sr),
        srgb_channel_to_linear(sg),
        srgb_channel_to_linear(sb),
    )


def kelvin_to_mood(kelvin: float) -> str:
    """
    Returns concise, high-signal photographic mood token for diffusion models.
    Avoids prompt pollution with raw numeric numbers.
    """
    k = float(kelvin)
    if k < 2500:
        return "deep candle amber warmth"
    elif k < 3400:
        return "warm tungsten"
    elif k < 4500:
        return "warm studio halogen"
    elif k < 5800:
        return "neutral daylight strobe"
    elif k < 7000:
        return "cool overcast daylight"
    elif k < 9000:
        return "cinematic cool blue"
    else:
        return "deep twilight cyan"


MODIFIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Octabox (Soft)": {
        "token": "soft diffused octabox light, gentle shadow falloff",
        "wrap_factor": 0.38,
        "specular_power": 16.0,
        "specular_strength": 0.20,
        "falloff_exponent": 1.0,
        "aperture_size": 1.40,
    },
    "Beauty Dish (Silver)": {
        "token": "punchy beauty dish lighting, crisp round catchlights",
        "wrap_factor": 0.16,
        "specular_power": 45.0,
        "specular_strength": 0.45,
        "falloff_exponent": 1.4,
        "aperture_size": 0.55,
    },
    "Fresnel Spotlight": {
        "token": "hard optical fresnel beam, high-contrast chiaroscuro shadows",
        "wrap_factor": 0.04,
        "specular_power": 85.0,
        "specular_strength": 0.70,
        "falloff_exponent": 2.0,
        "aperture_size": 0.20,
    },
    "Hard Snoot": {
        "token": "tight snoot spotlight, dramatic isolation pool of light",
        "wrap_factor": 0.0,
        "specular_power": 120.0,
        "specular_strength": 0.85,
        "falloff_exponent": 2.8,
        "aperture_size": 0.10,
    },
    "Direct On-Camera Flash": {
        "token": "raw direct on-camera flash, hard snapshot micro-shadows, glossy specularity",
        "wrap_factor": 0.02,
        "specular_power": 90.0,
        "specular_strength": 0.90,
        "falloff_exponent": 1.8,
        "aperture_size": 0.04,
    },
    "Stripbox": {
        "token": "narrow stripbox kicker, razor edge rim separation",
        "wrap_factor": 0.12,
        "specular_power": 60.0,
        "specular_strength": 0.55,
        "falloff_exponent": 1.6,
        "aperture_size": 0.30,
    },
}
