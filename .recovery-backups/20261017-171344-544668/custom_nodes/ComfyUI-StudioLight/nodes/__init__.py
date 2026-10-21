import importlib
import logging
import traceback

logger = logging.getLogger("ComfyUI-StudioLight")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_MODULES = (
    ".studio_light",
    ".si_shadow_highlight",
    ".photometric_composite",
    ".prompt_sanitizer",
    ".post_composite",
    ".nodes_transport",
    ".appearance_repair",
)

for _mod_name in _MODULES:
    try:
        _mod = importlib.import_module(_mod_name, package=__name__)
        NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
        NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    except Exception as _exc:
        logger.error(
            "\n======================================================================\n"
            "[ComfyUI-StudioLight] ERROR: Failed to load node module 'nodes%s' (%s).\n"
            "Other independent StudioLight modules will continue loading.\n"
            "Full traceback:\n%s"
            "======================================================================",
            _mod_name,
            _exc,
            traceback.format_exc(),
        )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


