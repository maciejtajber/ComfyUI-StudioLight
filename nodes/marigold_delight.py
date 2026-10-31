"""Experimental Marigold IID Lighting adapter. Original adapter code: MIT.

External Diffusers code and model weights retain their respective licenses.
No weights are shipped with this module; imports and loading are lazy.
"""
import logging
from pathlib import Path

import torch

MODEL_ID = "prs-eth/marigold-iid-lighting-v1-1"
log = logging.getLogger("ComfyUI-StudioLight")


def linear_to_srgb(value):
    value = value.clamp(min=0)
    return torch.where(value <= 0.0031308, 12.92 * value,
                       1.055 * value.pow(1.0 / 2.4) - 0.055)


def _device(choice):
    if choice != "auto":
        device = torch.device(choice)
    else:
        try:
            import comfy.model_management as mm
            device = mm.get_torch_device()
        except ImportError:
            device = torch.device("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available in this Python environment.")
    return device


def _validate_pipeline(pipe):
    props = pipe.config.target_properties
    names = list(props["target_names"])
    if set(names) != {"albedo", "shading", "residual"} or len(names) != 3:
        raise ValueError("Expected IID Lighting (albedo, shading, residual), not depth or IID Appearance.")
    if any(props[name]["prediction_space"] != "linear" for name in names):
        raise ValueError("Expected linear-space IID Lighting outputs.")
    return names


class StudioLightMarigoldDeLight:
    """Keep raw linear maps in a typed socket; IMAGE outputs are sRGB previews."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            "processing_resolution": ("INT", {"default": 768, "min": 128, "max": 1536, "step": 8}),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            "device": (["auto", "cuda", "mps", "cpu"],),
            "allow_download": ("BOOLEAN", {"default": False,
                "tooltip": "Enable for the first run to download the pretrained model. Weights have their own license."}),
            "offload_after_run": ("BOOLEAN", {"default": True}),
            "local_model_path": ("STRING", {"default": "",
                "tooltip": "Optional local Diffusers model directory containing model_index.json. Empty uses the official model cache."}),
        }}

    RETURN_TYPES = ("STUDIOLIGHT_INTRINSICS", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("intrinsics_linear", "albedo_srgb", "shading_preview_srgb",
                    "residual_preview_srgb", "uncalibrated_reconstruction_srgb", "report")
    FUNCTION = "decompose"
    CATEGORY = "StudioLight/Experimental"
    DESCRIPTION = "Estimate albedo, old shading and residual. This does not apply new lights. IMAGE previews use sRGB."

    def __init__(self):
        self._pipe = None
        self._key = None

    def _load(self, local_model_path, allow_download, device):
        try:
            from diffusers import MarigoldIntrinsicsPipeline
        except ImportError as exc:
            raise RuntimeError("Install requirements-marigold.txt in ComfyUI's Python environment, then restart ComfyUI.") from exc
        source = str(Path(local_model_path).expanduser().resolve()) if local_model_path.strip() else MODEL_ID
        if local_model_path.strip() and not (Path(source) / "model_index.json").is_file():
            raise ValueError("local_model_path must contain a Diffusers model_index.json file.")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        key = (source, dtype)
        if key != self._key:
            self._pipe = None
            self._key = None
            options = {"torch_dtype": dtype, "local_files_only": not allow_download}
            try:
                import folder_paths
                options["cache_dir"] = str(Path(folder_paths.models_dir) / "studiolight" / "huggingface")
            except ImportError:
                pass
            try:
                pipe = MarigoldIntrinsicsPipeline.from_pretrained(source, **options)
            except OSError as exc:
                raise RuntimeError("Marigold could not load. For first use enable allow_download with internet access, or provide a complete local model directory. " + str(exc)) from exc
            _validate_pipeline(pipe)
            pipe.enable_attention_slicing()
            pipe.vae.enable_slicing()
            self._pipe, self._key = pipe, key
        if device.type != "cpu":
            try:
                import comfy.model_management as mm
                mm.unload_all_models()
                mm.soft_empty_cache()
            except ImportError:
                pass
        self._pipe.to(device)
        return self._pipe

    @torch.inference_mode()
    def decompose(self, image, seed=0, steps=4, processing_resolution=768,
                  ensemble_size=1, device="auto", allow_download=False,
                  offload_after_run=True, local_model_path=""):
        if image.ndim != 4 or image.shape[-1] != 3 or image.shape[0] < 1:
            raise ValueError("Expected a nonempty ComfyUI IMAGE batch shaped [B,H,W,3].")
        if image.shape[1] < 1 or image.shape[2] < 1 or not torch.isfinite(image).all():
            raise ValueError("Input image must have positive dimensions and finite pixel values.")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Input must be sRGB in [0,1], not linear HDR data.")
        chosen_device = _device(device)
        pipe = self._load(local_model_path, allow_download, chosen_device)
        names = _validate_pipeline(pipe)
        outputs = []
        try:
            for index, frame in enumerate(image):
                # CPU generators also work with Diffusers on MPS; avoid changing global RNG state.
                rng = torch.Generator(device="cpu").manual_seed((seed + index) % (2**64))
                prediction = pipe(
                    image=frame.detach().to("cpu", dtype=torch.float32).permute(2, 0, 1).unsqueeze(0),
                    num_inference_steps=steps, ensemble_size=ensemble_size,
                    processing_resolution=processing_resolution, match_input_resolution=True,
                    batch_size=1, generator=rng, output_type="pt",
                ).prediction
                expected = (3, 3, image.shape[1], image.shape[2])
                if tuple(prediction.shape) != expected or not torch.isfinite(prediction).all():
                    raise RuntimeError("Unexpected/nonfinite Marigold prediction: " + str(tuple(prediction.shape)))
                outputs.append(prediction.detach().to("cpu", dtype=torch.float32).permute(0, 2, 3, 1).contiguous())
        finally:
            if offload_after_run:
                pipe.to("cpu")
                if chosen_device.type != "cpu":
                    try:
                        import comfy.model_management as mm
                        mm.soft_empty_cache()
                    except ImportError:
                        pass
        stacked = torch.stack(outputs)
        maps = {name: stacked[:, index].contiguous() for index, name in enumerate(names)}
        reconstruction = maps["albedo"] * maps["shading"] + maps["residual"]
        bundle = {**maps, "color_space": "linear", "model_id": MODEL_ID,
                  "source": local_model_path.strip() or MODEL_ID,
                  "shading_up_to_scale": True, "residual_up_to_scale": True,
                  "seed": seed, "steps": steps, "ensemble_size": ensemble_size,
                  "processing_resolution": processing_resolution}
        previews = [linear_to_srgb(maps[name]).clamp(0, 1) for name in ("albedo", "shading", "residual")]
        report = (f"Marigold IID Lighting | device={chosen_device} | steps={steps} | ensemble={ensemble_size}\n"
                  "Raw maps: linear float32 CPU tensors, original input dimensions. IMAGE outputs: sRGB.\n"
                  "Shading and residual are up to scale. A*S+R is UNCALIBRATED, not a reconstruction accuracy metric.\n"
                  "Old residual is NOT added to the albedo. Preview resizing cannot recover missing fine detail.")
        return (bundle, *previews, linear_to_srgb(reconstruction).clamp(0, 1), report)


NODE_CLASS_MAPPINGS = {"StudioLightMarigoldDeLight": StudioLightMarigoldDeLight}
NODE_DISPLAY_NAME_MAPPINGS = {"StudioLightMarigoldDeLight": "StudioLight De-Light — Marigold (Experimental)"}
