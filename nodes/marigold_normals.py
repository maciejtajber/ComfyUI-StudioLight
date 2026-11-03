"""Marigold Surface Normals adapter for ComfyUI-StudioLight.
Original adapter code: Copyright 2026 Maciej Tajber (MIT License).

Upstream model weights (prs-eth/marigold-normals-v1-1) are downloaded on first
use from Hugging Face and remain governed by CreativeML Open RAIL++-M.
"""
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_ID = "prs-eth/marigold-normals-v1-1"
REVISION = "09cfdd258cb281fa006cf1afcd2284376d16687d"


def encode_normals(prediction, convention):
    """Encode signed BCHW normals as BHWC data; no sRGB transfer function."""
    n = prediction.detach().float()
    if n.ndim != 4 or n.shape[1] != 3 or not torch.isfinite(n).all():
        raise ValueError("Expected finite BCHW normals with three channels")
    n = F.normalize(n, dim=1, eps=1e-6)
    if convention == "dsine_compatible":
        # Marigold: X right, Y up, front-facing normal Z=+1.
        # DSINE: X right, Y down, front-facing normal Z=+1.
        n = n * n.new_tensor([1, -1, 1]).view(1, 3, 1, 1)
    elif convention != "marigold_native":
        raise ValueError("Unknown normal convention")
    return ((n.permute(0, 2, 3, 1) + 1) * 0.5).clamp(0, 1).cpu()


class StudioLightMarigoldNormals:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**63-1,
                              "control_after_generate": True}),
            "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
            "processing_resolution": ("INT", {"default": 768, "min": 64,
                                               "max": 2048, "step": 64}),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 10}),
            "output_convention": (["marigold_native", "dsine_compatible"],),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("normals", "native_preview", "report")
    FUNCTION = "estimate"
    CATEGORY = "StudioLight/Geometry"

    @torch.inference_mode()
    def estimate(self, image, seed, steps, processing_resolution,
                 ensemble_size, output_convention):
        import comfy.model_management as mm
        import folder_paths
        try:
            from diffusers import MarigoldNormalsPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Missing Diffusers dependency for Marigold Normals. "
                "Run `pip install -r requirements.txt` in ComfyUI's Python environment, then restart ComfyUI."
            ) from exc

        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Expected BHWC RGB IMAGE")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("IMAGE must be finite sRGB in [0,1]")
        device = mm.get_torch_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        cache = Path(folder_paths.models_dir) / "studiolight" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        snapshot = cache / "models--prs-eth--marigold-normals-v1-1" / "snapshots" / REVISION
        has_fp16 = (snapshot / "unet/diffusion_pytorch_model.fp16.safetensors").is_file()
        weights_suffix = ".fp16.safetensors" if has_fp16 else ".safetensors"
        required = ["model_index.json", "unet/config.json",
                    f"unet/diffusion_pytorch_model{weights_suffix}",
                    "text_encoder/config.json", f"text_encoder/model{weights_suffix}",
                    "vae/config.json", f"vae/diffusion_pytorch_model{weights_suffix}",
                    "scheduler/scheduler_config.json", "tokenizer/vocab.json",
                    "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
                    "tokenizer/special_tokens_map.json"]
        missing = [name for name in required if not (snapshot / name).is_file()]
        mm.unload_all_models()
        mm.soft_empty_cache()
        pipeline = None
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                if not missing:
                    kwargs = {
                        "torch_dtype": dtype,
                        "cache_dir": str(cache),
                        "local_files_only": True,
                    }
                    if has_fp16:
                        kwargs["variant"] = "fp16"
                    pipeline = MarigoldNormalsPipeline.from_pretrained(
                        str(snapshot), **kwargs
                    ).to(device)
                else:
                    # First-run automatic download from Hugging Face into StudioLight cache
                    dl_kwargs = {
                        "revision": REVISION,
                        "torch_dtype": dtype,
                        "cache_dir": str(cache),
                    }
                    if dtype == torch.float16:
                        try:
                            pipeline = MarigoldNormalsPipeline.from_pretrained(
                                MODEL_ID, variant="fp16", **dl_kwargs
                            ).to(device)
                        except OSError:
                            pipeline = MarigoldNormalsPipeline.from_pretrained(
                                MODEL_ID, **dl_kwargs
                            ).to(device)
                    else:
                        pipeline = MarigoldNormalsPipeline.from_pretrained(
                            MODEL_ID, **dl_kwargs
                        ).to(device)
                source = image.detach().float().cpu().permute(0, 3, 1, 2)
                result = pipeline(
                    image=source, num_inference_steps=steps,
                    processing_resolution=processing_resolution,
                    ensemble_size=ensemble_size,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    match_input_resolution=True, output_type="pt",
                )
                prediction = result.prediction
                oriented = encode_normals(prediction, output_convention)
                native = encode_normals(prediction, "marigold_native")
        finally:
            if pipeline is not None:
                pipeline.to("cpu")
                del pipeline
            gc.collect()
            mm.soft_empty_cache()
        report = (f"Marigold Normals: {MODEL_ID}@{REVISION[:8]}; "
                  f"steps={steps}, processing={processing_resolution}, "
                  f"ensemble={ensemble_size}; convention={output_convention}. "
                  "Linear unit normals encoded in [0,1]; no gamma transfer function.")
        return oriented, native, report


# Keep StudioLightMarigoldNormalsTest alias for backward compatibility with saved workflows
StudioLightMarigoldNormalsTest = StudioLightMarigoldNormals

NODE_CLASS_MAPPINGS = {
    "StudioLightMarigoldNormals": StudioLightMarigoldNormals,
    "StudioLightMarigoldNormalsTest": StudioLightMarigoldNormalsTest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightMarigoldNormals": "StudioLight Marigold Normals",
    "StudioLightMarigoldNormalsTest": "StudioLight Marigold Normals",
}
