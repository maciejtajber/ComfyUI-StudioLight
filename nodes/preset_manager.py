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


def list_presets() -> List[str]:
    """Returns sorted list of preset names found in presets/ directory, starting with 'Custom'."""
    names = ["Custom"]
    if os.path.exists(PRESETS_DIR):
        for f in sorted(os.listdir(PRESETS_DIR)):
            if f.endswith(".json"):
                path = os.path.join(PRESETS_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        name = data.get("name", os.path.splitext(f)[0])
                        names.append(name)
                except Exception:
                    continue
    return names


def load_preset_data(preset_name: str) -> Dict[str, Any]:
    """Loads preset JSON data by display name. Returns empty dict if 'Custom' or not found."""
    if preset_name == "Custom" or not os.path.exists(PRESETS_DIR):
        return {}

    for f in os.listdir(PRESETS_DIR):
        if f.endswith(".json"):
            path = os.path.join(PRESETS_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if data.get("name") == preset_name or os.path.splitext(f)[0] == preset_name:
                        return data
            except Exception:
                continue
    return {}
