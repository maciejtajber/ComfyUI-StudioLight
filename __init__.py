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

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


