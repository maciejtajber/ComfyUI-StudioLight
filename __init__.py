"""
ComfyUI-StudioLight
Professional Photographic & Fashion Studio Lighting Controller for ComfyUI.
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT
"""

import logging

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
    logger.warning(
        "[ComfyUI-StudioLight] Failed to initialize node mappings gracefully (server startup continues): %s",
        exc,
    )

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

