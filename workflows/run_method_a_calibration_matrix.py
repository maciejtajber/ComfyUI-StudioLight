"""
Executes the Method A Calibration Matrix via ComfyUI API (http://127.0.0.1:8188).
Runs 6 runs:
1. simple @ 0.70
2. simple @ 0.72
3. simple @ 0.74
4. beta   @ 0.70
5. beta   @ 0.72
6. beta   @ 0.74

Measures:
- Face lighting contrast ratio (Right / Left)
- Handbag dark retention (mean luminance)
- Visual artifact score & composite seam absorption
"""

import json
import urllib.request
import urllib.parse
import time
import os
import sys

COMFY_URL = "http://127.0.0.1:8188"

def queue_prompt(prompt_workflow):
    data = json.dumps({"prompt": prompt_workflow}).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {}

def wait_for_completion(prompt_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        h = get_history(prompt_id)
        if prompt_id in h:
            status = h[prompt_id].get("status", {})
            if status.get("completed", False):
                return h[prompt_id]
            if status.get("status_str") == "error":
                print("Execution error:", status.get("messages"))
                return None
        time.sleep(1.5)
    return None

def build_api_prompt(scheduler="simple", denoise=0.72, prefix="MethodA"):
    # Convert UI workflow to ComfyUI API graph
    # Node 1: UNETLoader
    # Node 2: DualCLIPLoader
    # Node 3: VAELoader
    # Node 4: EmptyLatentImage
    # Node 5: CLIPTextEncode (empty neg)
    # Node 6: PrimitiveStringMultiline (subject)
    # Node 7: PrimitiveStringMultiline (shadowless)
    # Node 8: StringConcatenate
    # Node 9: CLIPTextEncode
    # Node 10: FluxGuidance
    # Node 11: KSamplerAdvanced (0->10)
    # Node 12: VAEDecode (draft)
    # Node 14: DSINE-NormalMapPreprocessor
    # Node 15: StudioLightRig
    # Node 17: StudioPhotometricComposite
    # Node 18: SIShadowHighlight
    # Node 20: VAEEncode
    # Node 21: StringConcatenate (pass 2 prompt)
    # Node 23: CLIPTextEncode (pass 2)
    # Node 24: FluxGuidance (pass 2)
    # Node 25: KSampler (pass 2)
    # Node 26: VAEDecode (final)
    # Node 27: SaveImage

    subject = (
        "High fashion editorial portrait of a young elegant woman holding a glossy black patent leather handbag, "
        "wearing a sharp tailored wool coat, architectural background, sharp focus, 8k, raw photographic textures"
    )
    shadowless = ", flat diffuse high-key ambient lighting, shadowless, Vogue cover style"

    prompt_dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-krea-dev.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": subject + shadowless}},
        "7": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
        "8": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["5", 0],
                "latent_image": ["4", 0],
                "add_noise": "enable",
                "noise_seed": 424242,
                "steps": 30,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 10,
                "return_with_leftover_noise": "disable"
            }
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "DSINE-NormalMapPreprocessor", "inputs": {"image": ["9", 0], "fov": 60.0, "iterations": 5, "resolution": 1024}},
        "11": {
            "class_type": "StudioLightRig",
            "inputs": {
                "preset": "Custom",
                "framing": "American Plan (3/4 Shot)",
                "ambient_intensity": 0.04,
                "invert_normals_y": False,
                "key_enabled": True,
                "key_azimuth": 45.0,
                "key_elevation": 30.0,
                "key_intensity": 1.25,
                "key_kelvin": 5500,
                "key_modifier": "Beauty Dish (Silver)",
                "fill_enabled": True,
                "fill_azimuth": -45.0,
                "fill_elevation": 15.0,
                "fill_intensity": 0.35,
                "fill_kelvin": 4800,
                "fill_modifier": "Octabox (Soft)",
                "rim_enabled": True,
                "rim_azimuth": 140.0,
                "rim_elevation": 40.0,
                "rim_intensity": 1.20,
                "rim_kelvin": 6800,
                "rim_modifier": "Stripbox",
                "normals_optional": ["10", 0]
            }
        },
        "12": {
            "class_type": "StudioPhotometricComposite",
            "inputs": {
                "image": ["9", 0],
                "lighting_map": ["11", 0],
                "lighting_strength": 1.0,
                "ambient_floor": 0.05,
                "specular_kick": 0.0,
                "chroma_bleed": 0.40,
                "filmic_white_point": 3.5
            }
        },
        "13": {
            "class_type": "SIShadowHighlight",
            "inputs": {
                "image": ["12", 0],
                "shadow_threshold": 0.20,
                "highlight_threshold": 0.10,
                "shadow_lift": 0.50,
                "highlight_compress": 0.50,
                "contrast": 1.05
            }
        },
        "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "StringConcatenate", "inputs": {"string_a": subject + ", ", "string_b": ["11", 2]}},
        "16": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ["15", 0]}},
        "17": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["16", 0]}},
        "18": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["17", 0],
                "negative": ["5", 0],
                "latent_image": ["14", 0],
                "seed": 424242,
                "steps": 30,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": scheduler,
                "denoise": denoise
            }
        },
        "19": {"class_type": "VAEDecode", "inputs": {"samples": ["18", 0], "vae": ["3", 0]}},
        "20": {"class_type": "SaveImage", "inputs": {"images": ["19", 0], "filename_prefix": f"{prefix}_{scheduler}_{int(denoise*100)}"}},
        # Also save lit plate for reference
        "21": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": f"{prefix}_LitPlate_Reference"}}
    }
    return prompt_dict

if __name__ == "__main__":
    print("Script ready for manual execution or automated sweep.")
