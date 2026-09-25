"""
ComfyUI-StudioLight
Professional Photographic & Fashion Studio Lighting Controller for ComfyUI.
Author: Maciej Tajber (https://github.com/maciejtajber)
License: MIT
"""

from .nodes.studio_light import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
