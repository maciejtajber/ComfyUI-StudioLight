"""
StudioLight Trajectory Nodes — Method B: True Trajectory Injection
Implements single-pass continuous latent trajectory splitting and delta-injection
for FLUX flow-matching and diffusion models in ComfyUI.
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


class StudioLightTrajectorySplitter:
    """
    Pass 1: Executes the diffusion trajectory from step 0 to split_step in a single pass.
    Intercepts the model's un-overshot clean prediction (x0) via the native step callback,
    and returns both the clean geometry (clean_x0) and continuous noisy state (noisy_xt).
    Eliminates duplicate samplers and prevents overshooting artifacts.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "split_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("clean_x0", "noisy_xt")
    FUNCTION = "split_sample"
    CATEGORY = "StudioLight/Trajectory"

    def split_sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        split_step,
        cfg,
        sampler_name,
        scheduler,
        denoise=1.0,
    ):
        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)

        batch_inds = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)

        noise_mask = latent_image.get("noise_mask", None)

        preview_callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        captured_denoised = None

        def intercept_callback(step, denoised, x, total_steps):
            nonlocal captured_denoised
            # Capture the latest clean prediction x0 from Euler before stepping
            captured_denoised = denoised.detach().clone()
            if preview_callback is not None:
                preview_callback(step, denoised, x, total_steps)

        # Execute 0 -> split_step with leftover noise enabled (force_full_denoise=False)
        # This preserves the exact mathematical trajectory without zero-sigma leap
        samples = comfy.sample.sample(
            model,
            noise,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_samples,
            denoise=denoise,
            disable_noise=False,
            start_step=0,
            last_step=split_step,
            force_full_denoise=False,
            noise_mask=noise_mask,
            callback=intercept_callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        # Pack noisy state (L_10 in ComfyUI scaled latent space)
        noisy_xt = latent_image.copy()
        noisy_xt["samples"] = samples

        # Pack clean prediction (x0 in standard VAE-decodeable latent space)
        clean_x0 = latent_image.copy()
        if captured_denoised is not None:
            # Process from internal diffusion space to standard VAE latent space
            if hasattr(model.model, "process_latent_out"):
                clean_samples = model.model.process_latent_out(captured_denoised.to(torch.float32))
            else:
                clean_samples = captured_denoised.to(torch.float32)
            clean_x0["samples"] = clean_samples.to(samples.device)
        else:
            clean_x0["samples"] = samples

        return (clean_x0, noisy_xt)


class StudioLightTrajectoryInject:
    """
    Recombines the relit clean latent (relit_x0) with the preserved continuous
    noisy trajectory (noisy_xt) using the algebraic delta:
        injected = noisy_xt + (relit_x0 - clean_x0) * strength
    When passed to KSamplerAdvanced (add_noise=disable), ComfyUI's CONST noise_scaling
    mathematically scales this delta into the exact target state x'_t at machine precision.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noisy_xt": ("LATENT",),
                "clean_x0": ("LATENT",),
                "relit_x0": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("injected_latent",)
    FUNCTION = "inject"
    CATEGORY = "StudioLight/Trajectory"

    def inject(self, noisy_xt, clean_x0, relit_x0, strength=1.0):
        if not isinstance(noisy_xt, dict) or "samples" not in noisy_xt:
            raise ValueError("StudioLightTrajectoryInject: noisy_xt must be a valid LATENT dict with 'samples'")
        if not isinstance(clean_x0, dict) or "samples" not in clean_x0:
            raise ValueError("StudioLightTrajectoryInject: clean_x0 must be a valid LATENT dict with 'samples'")
        if not isinstance(relit_x0, dict) or "samples" not in relit_x0:
            raise ValueError("StudioLightTrajectoryInject: relit_x0 must be a valid LATENT dict with 'samples'")

        noisy_samples = noisy_xt["samples"]
        target_device = noisy_samples.device
        target_dtype = noisy_samples.dtype

        clean_samples = clean_x0["samples"].to(device=target_device, dtype=target_dtype)
        relit_samples = relit_x0["samples"].to(device=target_device, dtype=target_dtype)

        if clean_samples.shape != noisy_samples.shape or relit_samples.shape != noisy_samples.shape:
            raise ValueError(
                f"StudioLightTrajectoryInject: Shape mismatch. noisy={noisy_samples.shape}, "
                f"clean={clean_samples.shape}, relit={relit_samples.shape}."
            )

        # Delta represents pure photometric lighting change
        delta = (relit_samples - clean_samples) * float(strength)
        injected_samples = noisy_samples + delta

        out = noisy_xt.copy()
        out["samples"] = injected_samples
        return (out,)


NODE_CLASS_MAPPINGS = {
    "StudioLightTrajectorySplitter": StudioLightTrajectorySplitter,
    "StudioLightTrajectoryInject": StudioLightTrajectoryInject,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioLightTrajectorySplitter": "StudioLight Trajectory Splitter (Pass 1)",
    "StudioLightTrajectoryInject": "StudioLight Trajectory Delta Inject",
}
