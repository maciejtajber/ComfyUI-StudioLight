"""
ComfyUI-StudioLight — Preset Manager
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT

Discovers and loads lighting preset JSON files from the presets/ directory.
"""

import os
import json
from typing import Dict, Any, List

PRESETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "presets"))


_PRESET_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_all_presets() -> Dict[str, Dict[str, Any]]:
    global _PRESET_CACHE
    if not os.path.exists(PRESETS_DIR):
        return {}

    cache = {}
    for f in sorted(os.listdir(PRESETS_DIR)):
        if f.endswith(".json"):
            path = os.path.join(PRESETS_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    name = data.get("name", os.path.splitext(f)[0])
                    cache[name] = data
            except Exception:
                continue
    _PRESET_CACHE = cache
    return cache


def list_presets() -> List[str]:
    """Returns sorted list of preset names found in presets/ directory, starting with 'Custom'."""
    if not _PRESET_CACHE:
        _load_all_presets()
    names = ["Custom"] + sorted(list(_PRESET_CACHE.keys()))
    return names


def load_preset_data(preset_name: str) -> Dict[str, Any]:
    """Loads preset JSON data by display name. Returns empty dict if 'Custom' or not found."""
    if preset_name == "Custom":
        return {}
    if not _PRESET_CACHE:
        _load_all_presets()
    return _PRESET_CACHE.get(preset_name, {})


def get_all_presets_flattened() -> Dict[str, Dict[str, Any]]:
    """
    Returns all presets flattened into LiteGraph widget key/value dictionaries.
    Ensures frontend JavaScript receives the exact single-source-of-truth from JSON.
    """
    if not _PRESET_CACHE:
        _load_all_presets()

    flattened = {}
    for name, p in _PRESET_CACHE.items():
        k = p.get("key", {})
        f = p.get("fill", {})
        r = p.get("rim", {})
        flattened[name] = {
            "ambient_intensity": p.get("ambient_intensity", 0.05),
            "key_enabled": k.get("enabled", True),
            "key_azimuth": k.get("azimuth", 45.0),
            "key_elevation": k.get("elevation", 35.0),
            "key_distance": k.get("distance", 1.8),
            "key_intensity": k.get("intensity", 1.0),
            "key_kelvin": k.get("kelvin", 5500),
            "key_modifier": k.get("modifier", "Octabox (Soft)"),
            "fill_enabled": f.get("enabled", True),
            "fill_azimuth": f.get("azimuth", -30.0),
            "fill_elevation": f.get("elevation", 15.0),
            "fill_distance": f.get("distance", 2.5),
            "fill_intensity": f.get("intensity", 0.35),
            "fill_kelvin": f.get("kelvin", 4800),
            "fill_modifier": f.get("modifier", "Octabox (Soft)"),
            "rim_enabled": r.get("enabled", True),
            "rim_azimuth": r.get("azimuth", 140.0),
            "rim_elevation": r.get("elevation", 40.0),
            "rim_distance": r.get("distance", 1.5),
            "rim_intensity": r.get("intensity", 0.90),
            "rim_kelvin": r.get("kelvin", 6500),
            "rim_modifier": r.get("modifier", "Stripbox"),
            "normal_boost": p.get("normal_boost", 2.50),
            "ao_anchoring": p.get("ao_anchoring", 0.65),
        }
    return flattened


# Register ComfyUI Server API route so frontend can fetch presets dynamically
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/studiolight/presets")
    async def get_studiolight_presets_route(request):
        return web.json_response(get_all_presets_flattened())
except Exception:
    pass
