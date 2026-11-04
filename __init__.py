"""
ComfyUI-StudioLight
Professional Photographic & Fashion Studio Lighting Controller for ComfyUI.
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT
"""

import logging
import traceback

logger = logging.getLogger("ComfyUI-StudioLight")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import (
        NODE_CLASS_MAPPINGS as _NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _NODE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as exc:
    logger.error(
        "\n======================================================================\n"
        "[ComfyUI-StudioLight] ERROR: Failed to initialize node mappings (%s).\n"
        "ComfyUI server startup will continue, but StudioLight nodes may be unavailable.\n"
        "Full traceback:\n%s"
        "======================================================================",
        exc,
        traceback.format_exc(),
    )

# Auto-provision example portrait to ComfyUI input directory if missing
try:
    import os
    import shutil
    import folder_paths

    input_dir = folder_paths.get_input_directory()
    if input_dir and os.path.isdir(input_dir):
        sample_img = "StudioLight_T2I_00003_.png"
        sample_src = os.path.join(os.path.dirname(__file__), "assets", sample_img)
        sample_dst = os.path.join(input_dir, sample_img)
        if os.path.exists(sample_src) and not os.path.exists(sample_dst):
            shutil.copy2(sample_src, sample_dst)
except Exception:
    pass

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


