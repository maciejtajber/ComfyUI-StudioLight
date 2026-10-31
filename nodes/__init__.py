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

from .subject_background_base import NODE_CLASS_MAPPINGS as _sb_map, NODE_DISPLAY_NAME_MAPPINGS as _sb_disp
NODE_CLASS_MAPPINGS.update(_sb_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_sb_disp)

from .consistent_relighting import NODE_CLASS_MAPPINGS as _cr_map, NODE_DISPLAY_NAME_MAPPINGS as _cr_disp
NODE_CLASS_MAPPINGS.update(_cr_map)
NODE_DISPLAY_NAME_MAPPINGS.update(_cr_disp)

try:
    from .marigold_delight import NODE_CLASS_MAPPINGS as _md_map, NODE_DISPLAY_NAME_MAPPINGS as _md_disp
    NODE_CLASS_MAPPINGS.update(_md_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_md_disp)
except Exception as e:
    logger.warning("Marigold De-Light node skipped: %s", e)

try:
    from .marigold_normals import NODE_CLASS_MAPPINGS as _mn_map, NODE_DISPLAY_NAME_MAPPINGS as _mn_disp
    NODE_CLASS_MAPPINGS.update(_mn_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_mn_disp)
except Exception as e:
    logger.warning("Marigold Normals node skipped: %s", e)

try:
    from .delight_strength import NODE_CLASS_MAPPINGS as _ds_map, NODE_DISPLAY_NAME_MAPPINGS as _ds_disp
    NODE_CLASS_MAPPINGS.update(_ds_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ds_disp)
except Exception as e:
    logger.warning("DeLight Strength node skipped: %s", e)

try:
    from .post_composite import NODE_CLASS_MAPPINGS as _post_map, NODE_DISPLAY_NAME_MAPPINGS as _post_disp
    NODE_CLASS_MAPPINGS.update(_post_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_post_disp)
except Exception:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
