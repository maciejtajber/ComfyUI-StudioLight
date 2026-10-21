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
        return "warm golden halogen"
    elif k < 5800:
        return "clean neutral daylight"
    elif k < 7000:
        return "cool overcast daylight"
    elif k < 9000:
        return "cinematic cool blue"
    else:
        return "deep twilight cyan"


MODIFIER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Octabox (Soft)": {
        "token": "wide-aperture soft diffused illumination, gentle shadow falloff",
        "geometry": "octagon",
        "radius_m": 0.70,          # 140 cm diameter
        "aspect_ratio": 1.0,
        "beam_angle_deg": 130.0,
        "edge_softness": 0.35,
        "wrap_factor": 0.38,
        "specular_power": 16.0,
        "specular_strength": 0.20,
        "specular_roughness": 0.38,
        "falloff_exponent": 1.0,
    },
    "Beauty Dish (Silver)": {
        "token": "punchy semi-specular directional light, crisp round catchlights",
        "geometry": "beauty_dish", # Parabolic dish with center deflector disk
        "radius_m": 0.275,         # 55 cm diameter
        "deflector_ratio": 0.36,   # Central deflector disc radius ratio
        "deflector_transmission": 0.12, # Center attenuation (not hard zero)
        "aspect_ratio": 1.0,
        "beam_angle_deg": 85.0,
        "edge_softness": 0.18,
        "wrap_factor": 0.16,
        "specular_power": 45.0,
        "specular_strength": 0.45,
        "specular_roughness": 0.18,
        "falloff_exponent": 1.4,
    },
    "Fresnel Spotlight": {
        "token": "collimated hard optical beam, high-contrast directional shadows",
        "geometry": "circle",
        "radius_m": 0.15,          # 30 cm optical lens
        "aspect_ratio": 1.0,
        "beam_angle_deg": 35.0,
        "edge_softness": 0.08,
        "wrap_factor": 0.04,
        "specular_power": 85.0,
        "specular_strength": 0.70,
        "specular_roughness": 0.08,
        "falloff_exponent": 2.0,
    },
    "Hard Snoot": {
        "token": "narrow-beam optical spotlight, dramatic isolation pool of light",
        "geometry": "circle",
        "radius_m": 0.05,          # 10 cm pin-spot aperture
        "aspect_ratio": 1.0,
        "beam_angle_deg": 18.0,
        "edge_softness": 0.04,
        "wrap_factor": 0.0,
        "specular_power": 120.0,
        "specular_strength": 0.85,
        "specular_roughness": 0.04,
        "falloff_exponent": 2.8,
    },
    "Direct On-Camera Flash": {
        "token": "crisp frontal point-source illumination, sharp micro-shadows, glossy specularity",
        "geometry": "rectangle",
        "width_m": 0.08,           # 8 cm × 5 cm compact strobe head
        "height_m": 0.05,
        "aspect_ratio": 1.6,
        "beam_angle_deg": 70.0,
        "edge_softness": 0.05,
        "wrap_factor": 0.02,
        "specular_power": 90.0,
        "specular_strength": 0.90,
        "specular_roughness": 0.03,
        "falloff_exponent": 1.8,
    },
    "Stripbox": {
        "token": "narrow linear grazing illumination, razor edge rim separation",
        "geometry": "rectangle",
        "width_m": 0.30,           # 30 cm × 120 cm (1:4 ratio)
        "height_m": 1.20,
        "aspect_ratio": 0.25,
        "beam_angle_deg": 65.0,
        "edge_softness": 0.15,
        "wrap_factor": 0.12,
        "specular_power": 60.0,
        "specular_strength": 0.55,
        "specular_roughness": 0.22,
        "falloff_exponent": 1.6,
    },
}

