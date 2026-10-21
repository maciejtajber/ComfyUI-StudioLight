import importlib
import logging
import traceback

logger = logging.getLogger("ComfyUI-StudioLight")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

from .studio_light import NODE_CLASS_MAPPINGS as _sl_map, NODE_DISPLAY_NAME_MAPPINGS as _sl_disp
NODE_CLASS_MAPPINGS.update(_sl_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_sl_disp)

from .si_shadow_highlight import NODE_CLASS_MAPPINGS as _sh_map, NODE_DISPLAY_NAME_MAPPINGS as _sh_disp
NODE_CLASS_MAPPINGS.update(_sh_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_sh_disp)

from .photometric_composite import NODE_CLASS_MAPPINGS as _pc_map, NODE_DISPLAY_NAME_MAPPINGS as _pc_disp
NODE_CLASS_MAPPINGS.update(_pc_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_pc_disp)

from .prompt_sanitizer import NODE_CLASS_MAPPINGS as _ps_map, NODE_DISPLAY_NAME_MAPPINGS as _ps_disp
NODE_CLASS_MAPPINGS.update(_ps_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_ps_disp)

try:
    from .post_composite import NODE_CLASS_MAPPINGS as _post_map, NODE_DISPLAY_NAME_MAPPINGS as _post_disp
    NODE_CLASS_MAPPINGS.update(_post_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_post_disp)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
