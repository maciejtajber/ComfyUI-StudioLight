#!/usr/bin/env python3
"""
Executes the Mid-Stream Injection Sweep workflow against the active ComfyUI server.
Monitors queue status, tracks progress, and downloads the output images to Mac Desktop.
"""

import json
import urllib.request
import urllib.error
import time
import os
import sys

COMFY_HOST = "127.0.0.1:8188"

prompt = {
    # Zone 1: Model Stack
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "flux1-krea-dev.safetensors",
            "weight_dtype": "default"
        }
    },
    "2": {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "t5xxl_fp16.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux"
        }
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "ae.safetensors"
        }
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["2", 0],
            "text": "A highly detailed mechanical rhinoceros beetle, iridescent metallic carapace with brass gears, resting on dark polished basalt stone, studio photography, sharp macro focus, flat high-key diffuse studio illumination, shadowless even ambient lighting"
        }
    },
    "5": {
        "class_type": "FluxGuidance",
        "inputs": {
            "guidance": 3.5,
            "conditioning": ["4", 0]
        }
    },
    "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        }
    },

    # Zone 2: Pass 1 Fast Geometry Draft (Steps 0 -> 10)
    "7": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "add_noise": "enable",
            "noise_seed": 424242,
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "start_at_step": 0,
            "end_at_step": 10,
            "return_with_leftover_noise": "disable"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["3", 0]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "Midstream_01_Draft"
        }
    },

    # Zone 3: 3D Surface Normals, Python Lighting & Photometric Composite
    "10": {
        "class_type": "DSINE-NormalMapPreprocessor",
        "inputs": {
            "image": ["8", 0],
            "fov": 60.0,
            "iterations": 5,
            "resolution": 1024
        }
    },
    "11": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["10", 0],
            "filename_prefix": "Midstream_02_Normals"
        }
    },
    "12": {
        "class_type": "StudioLightRig",
        "inputs": {
            "preset": "Rembrandt Classic",
            "framing": "Tabletop / Luxury (Perfume Bottle / Small Product)",
            "ambient_intensity": 0.04,
            "invert_normals_y": False,
            "key_enabled": True,
            "key_azimuth": 45.0,
            "key_elevation": 35.0,
            "key_intensity": 1.20,
            "key_kelvin": 5500,
            "key_modifier": "Octabox (Soft)",
            "fill_enabled": True,
            "fill_azimuth": -30.0,
            "fill_elevation": 15.0,
            "fill_intensity": 0.35,
            "fill_kelvin": 4800,
            "fill_modifier": "Octabox (Soft)",
            "rim_enabled": True,
            "rim_azimuth": 160.0,
            "rim_elevation": 45.0,
            "rim_intensity": 1.15,
            "rim_kelvin": 6500,
            "rim_modifier": "Stripbox",
            "normals_optional": ["10", 0]
        }
    },
    "13": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["12", 1],
            "filename_prefix": "Midstream_03_VFX_Probes"
        }
    },
    "14": {
        "class_type": "StudioPhotometricComposite",
        "inputs": {
            "image": ["8", 0],
            "lighting_map": ["12", 0],
            "lighting_strength": 1.0,
            "ambient_floor": 0.05,
            "specular_kick": 1.15,
            "chroma_bleed": 0.40,
            "filmic_white_point": 3.5
        }
    },
    "15": {
        "class_type": "SIShadowHighlight",
        "inputs": {
            "image": ["14", 0],
            "shadow_amount": 0.20,
            "highlight_amount": 0.10,
            "shadow_tonal_width": 0.50,
            "highlight_tonal_width": 0.50,
            "midtone_contrast": 1.05
        }
    },
    "16": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["15", 0],
            "filename_prefix": "Midstream_04_LitPlate"
        }
    },
    "17": {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["15", 0],
            "vae": ["3", 0]
        }
    },

    # Zone 4: The 4-Way Denoise Sweep
    # Run A: Denoise 0.75
    "20": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["17", 0],
            "seed": 424242,
            "control_after_generate": "fixed",
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.75
        }
    },
    "21": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["20", 0],
            "vae": ["3", 0]
        }
    },
    "22": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["21", 0],
            "filename_prefix": "Midstream_Sweep_RunA_Denoise075"
        }
    },

    # Run B: Denoise 0.55
    "23": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["17", 0],
            "seed": 424242,
            "control_after_generate": "fixed",
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.55
        }
    },
    "24": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["23", 0],
            "vae": ["3", 0]
        }
    },
    "25": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["24", 0],
            "filename_prefix": "Midstream_Sweep_RunB_Denoise055"
        }
    },

    # Run C: Denoise 0.40
    "26": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["17", 0],
            "seed": 424242,
            "control_after_generate": "fixed",
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.40
        }
    },
    "27": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["26", 0],
            "vae": ["3", 0]
        }
    },
    "28": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["27", 0],
            "filename_prefix": "Midstream_Sweep_RunC_Denoise040"
        }
    },

    # Run D: Denoise 0.25
    "29": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["5", 0],
            "negative": ["5", 0],
            "latent_image": ["17", 0],
            "seed": 424242,
            "control_after_generate": "fixed",
            "steps": 20,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 0.25
        }
    },
    "30": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["29", 0],
            "vae": ["3", 0]
        }
    },
    "31": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["30", 0],
            "filename_prefix": "Midstream_Sweep_RunD_Denoise025"
        }
    },
}


def queue_prompt(prompt_dict):
    data = json.dumps({"prompt": prompt_dict}).encode("utf-8")
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
        sys.exit(1)


if __name__ == "__main__":
    print(f">>> Submitting Mid-Stream Injection Sweep prompt to http://{COMFY_HOST}...")
    res = queue_prompt(prompt)
    prompt_id = res.get("prompt_id")
    print(f"✓ Prompt queued successfully! ID: {prompt_id}")
    print(f"  Node count: {len(prompt)}")
    print(f"  Tracking execution...")
