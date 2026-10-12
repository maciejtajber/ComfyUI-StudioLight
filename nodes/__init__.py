import importlib
import logging

logger = logging.getLogger("ComfyUI-StudioLight")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = (
    ".studio_light",
    ".si_shadow_highlight",
    ".photometric_composite",
    ".prompt_sanitizer",
)

for _mod_name in _MODULES:
    try:
        _mod = importlib.import_module(_mod_name, package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as _exc:
        logger.warning(
            "[ComfyUI-StudioLight] Skipping module '%s' due to import error: %s",
            _mod_name,
            _exc,
        )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

