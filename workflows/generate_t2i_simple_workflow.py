"""
Generates the clean, dedicated Text-to-Image workflow for ComfyUI-StudioLight:
Zone 1: Models & LoRA (Flux Krea Dev, Dual CLIP, VAE, ChromeCore LoRA)
Zone 2: Prompt & StudioLight Console (SIPromptBlock Subject + Universal Negative + StudioLightRig)
Zone 3: VFX Probes & Deferred Normal Pass (18% Grey & Chrome Ball + Light Radar + Normal Pass)
Zone 4: Flux Sampling & Master Output (FluxGuidance, Latent, KSampler, VAEDecode, SaveImage)
"""

import json
import os

nodes = []
links = []
link_counter = 1

def add_link(from_node, from_slot, to_node, to_slot, type_name):
    global link_counter
    l_id = link_counter
    link_counter += 1
    links.append([l_id, from_node, from_slot, to_node, to_slot, type_name])
    return l_id

# -------------------------------------------------------------
# 1. BASE MODELS (Flux Krea Dev, Dual CLIP, VAE, ChromeCore)
# -------------------------------------------------------------
# Node 1: UNETLoader
node_unet = {
    "id": 1,
    "type": "UNETLoader",
    "pos": [-1200, 100],
    "size": [320, 90],
    "flags": {},
    "order": 1,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "UNETLoader"},
    "widgets_values": ["flux1-krea-dev.safetensors", "default"],
    "title": "Flux Krea Dev (UNET)"
}
nodes.append(node_unet)

# Node 2: DualCLIPLoader
node_clip = {
    "id": 2,
    "type": "DualCLIPLoader",
    "pos": [-1200, 240],
    "size": [320, 110],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2, 3, 4], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "DualCLIPLoader"},
    "widgets_values": ["t5xxl_fp16.safetensors", "clip_l.safetensors", "flux"],
    "title": "Dual CLIP Loader (T5 + CLIP-L)"
}
nodes.append(node_clip)

# Node 3: VAELoader
node_vae = {
    "id": 3,
    "type": "VAELoader",
    "pos": [-1200, 400],
    "size": [320, 80],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "VAE", "type": "VAE", "links": [5], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAELoader"},
    "widgets_values": ["ae.safetensors"],
    "title": "Flux VAE"
}
nodes.append(node_vae)

# Node 4: LoRALoaderModelOnly
node_lora = {
    "id": 4,
    "type": "LoraLoaderModelOnly",
    "pos": [-840, 100],
    "size": [320, 90],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [6], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "LoraLoaderModelOnly"},
    "widgets_values": ["chromecore_flux.safetensors", 0.75],
    "title": "ChromeCore LoRA"
}
nodes.append(node_lora)
add_link(1, 0, 4, 0, "MODEL")

# -------------------------------------------------------------
# 2. PROMPT & STUDIOLIGHT CONSOLE
# -------------------------------------------------------------
# Node 5: SIPromptBlock (Subject)
node_pb_subject = {
    "id": 5,
    "type": "SIPromptBlock",
    "pos": [-480, 100],
    "size": [360, 280],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": [
        {"name": "clip", "type": "CLIP", "link": 2},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None},
        {"name": "mask", "type": "MASK", "link": None}
    ],
    "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [7], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "SIPromptBlock"},
    "widgets_values": [
        "Subject",
        "a high-fashion portrait bust of a feminine cyborg made of polished mirror chrome, sleek liquid metal surface, delicate facial features, intense gaze, neutral dark grey studio backdrop",
        True,
        1.0,
        "ADDITIVE",
        0.0,
        0.40,
        "concat"
    ],
    "title": "✨ [Subject] SI Prompt Block (0.0 → 0.40)"
}
nodes.append(node_pb_subject)
add_link(2, 0, 5, 0, "CLIP")

# Node 6: SIPromptBlock (Universal Negative)
node_pb_neg = {
    "id": 6,
    "type": "SIPromptBlock",
    "pos": [-480, 420],
    "size": [360, 280],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": [
        {"name": "clip", "type": "CLIP", "link": 3},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None},
        {"name": "mask", "type": "MASK", "link": None}
    ],
    "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [8], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "SIPromptBlock"},
    "widgets_values": [
        "Negative",
        "worst quality, blurry, noisy, low resolution, blown out light flares, deformed, extra limbs, bad anatomy, artifacts",
        True,
        1.0,
        "ADDITIVE",
        0.0,
        1.00,
        "concat"
    ],
    "title": "⛔ [Negative] Universal Suppression"
}
nodes.append(node_pb_neg)
add_link(2, 0, 6, 0, "CLIP")

# Node 7: StudioLightRig
node_studio_light = {
    "id": 7,
    "type": "StudioLightRig",
    "pos": [-80, 100],
    "size": [380, 720],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": [
        {"name": "normals_optional", "type": "IMAGE", "link": None},
        {"name": "depth_optional", "type": "IMAGE", "link": None},
        {"name": "clip_optional", "type": "CLIP", "link": 4},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": 7},
        {"name": "prompt_text_optional", "type": "STRING", "link": None}
    ],
    "outputs": [
        {"name": "relighting_map", "type": "IMAGE", "links": [9], "shape": 3, "slot_index": 0},
        {"name": "probe_preview", "type": "IMAGE", "links": [10], "shape": 3, "slot_index": 1},
        {"name": "conditioning", "type": "CONDITIONING", "links": [11], "shape": 3, "slot_index": 2},
        {"name": "lighting_prompt", "type": "STRING", "links": [], "shape": 3, "slot_index": 3}
    ],
    "properties": {"Node name for S&R": "StudioLightRig"},
    "widgets_values": [
        "Rembrandt Classic",
        "Portrait (Classical Bust)",
        0.05,
        False,
        True, 45.0, 35.0, 1.0, 5500, "Octabox (Soft)",
        True, -30.0, 15.0, 0.35, 4800, "Octabox (Soft)",
        True, 160.0, 45.0, 1.10, 6500, "Stripbox",
        0.40, 1.00, 1.00, "A_NORM"
    ],
    "title": "🎛️ StudioLight (Rembrandt Console)"
}
nodes.append(node_studio_light)
add_link(2, 0, 7, 2, "CLIP")
add_link(5, 0, 7, 3, "CONDITIONING")

# -------------------------------------------------------------
# 3. VFX PROBES & DEFERRED PASS PREVIEWS
# -------------------------------------------------------------
# Node 8: PreviewImage (VFX Probe 18% Grey & Chrome Ball + Radar)
node_probe_prev = {
    "id": 8,
    "type": "PreviewImage",
    "pos": [340, 100],
    "size": [320, 320],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 10}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "✨ VFX Probe (18% Grey & Chrome Ball + Radar)"
}
nodes.append(node_probe_prev)
add_link(7, 1, 8, 0, "IMAGE")

# Node 12: PreviewImage (32-bit Deferred Relighting Pass)
node_relight_prev = {
    "id": 12,
    "type": "PreviewImage",
    "pos": [340, 460],
    "size": [320, 320],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "✨ 32-bit Deferred Lighting Pass"
}
nodes.append(node_relight_prev)
add_link(7, 0, 12, 0, "IMAGE")

# -------------------------------------------------------------
# 4. FLUX SAMPLING & MASTER OUTPUT
# -------------------------------------------------------------
# Node 9: FluxGuidance
node_guidance = {
    "id": 9,
    "type": "FluxGuidance",
    "pos": [700, 100],
    "size": [260, 80],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 11}],
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [12], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "FluxGuidance"},
    "widgets_values": [3.5],
    "title": "Flux Guidance (3.5)"
}
nodes.append(node_guidance)
add_link(7, 2, 9, 0, "CONDITIONING")

# Node 10: EmptySD3LatentImage
node_latent = {
    "id": 10,
    "type": "EmptySD3LatentImage",
    "pos": [700, 220],
    "size": [260, 110],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [13], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptySD3LatentImage"},
    "widgets_values": [896, 1152, 1],
    "title": "Empty Latent (896x1152)"
}
nodes.append(node_latent)

# Node 11: KSampler
node_ksampler = {
    "id": 11,
    "type": "KSampler",
    "pos": [1000, 100],
    "size": [300, 260],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 6},
        {"name": "positive", "type": "CONDITIONING", "link": 12},
        {"name": "negative", "type": "CONDITIONING", "link": 8},
        {"name": "latent_image", "type": "LATENT", "link": 13}
    ],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [14], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [42, "randomize", 25, 1.0, "euler", "beta", 1.0],
    "title": "🚀 Flux KSampler"
}
nodes.append(node_ksampler)
add_link(4, 0, 11, 0, "MODEL")
add_link(9, 0, 11, 1, "CONDITIONING")
add_link(6, 0, 11, 2, "CONDITIONING")
add_link(10, 0, 11, 3, "LATENT")

# Node 13: VAEDecode
node_decode = {
    "id": 13,
    "type": "VAEDecode",
    "pos": [1000, 400],
    "size": [240, 80],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": [
        {"name": "samples", "type": "LATENT", "link": 14},
        {"name": "vae", "type": "VAE", "link": 5}
    ],
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [15], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": [],
    "title": "VAE Decode"
}
nodes.append(node_decode)
add_link(11, 0, 13, 0, "LATENT")
add_link(3, 0, 13, 1, "VAE")

# Node 14: SaveImage
node_save = {
    "id": 14,
    "type": "SaveImage",
    "pos": [1340, 100],
    "size": [420, 520],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 15}],
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["StudioLight_T2I"],
    "title": "🎉 [T2I] Master Render"
}
nodes.append(node_save)
add_link(13, 0, 14, 0, "IMAGE")

# Assemble final workflow dictionary
workflow_data = {
    "last_node_id": 14,
    "last_link_id": len(links),
    "nodes": nodes,
    "links": links,
    "groups": [
        {"title": "1. BASE MODELS (FLUX KREA DEV + CHROMECORE)", "bounding": [-1240, 40, 740, 480], "color": "#2a363b"},
        {"title": "2. PROMPT & STUDIOLIGHT CONSOLE", "bounding": [-520, 40, 840, 820], "color": "#3f2b47"},
        {"title": "3. VFX PROBES & DEFERRED PASS", "bounding": [300, 40, 380, 780], "color": "#1f3a3d"},
        {"title": "4. FLUX SAMPLING & 2K MASTER OUTPUT", "bounding": [660, 40, 1140, 620], "color": "#2e2d4d"}
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.85, "offset": [1300, 80]}},
    "version": 0.4
}

out_dir = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "StudioLight_T2I_Simple.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(workflow_data, f, indent=2)

print(f"Successfully generated clean T2I workflow at {out_path} with {len(nodes)} nodes and {len(links)} links!")

# Generate ComfyUI API Prompt format as well
api_prompt = {
    "1": {"inputs": {"unet_name": "flux1-krea-dev.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "2": {"inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}, "class_type": "DualCLIPLoader"},
    "3": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
    "4": {"inputs": {"lora_name": "chromecore_flux.safetensors", "strength_model": 0.75, "model": ["1", 0]}, "class_type": "LoraLoaderModelOnly"},
    "5": {
        "inputs": {
            "clip": ["2", 0],
            "label": "Subject",
            "text": "a high-fashion portrait bust of a feminine cyborg made of polished mirror chrome, sleek liquid metal surface, delicate facial features, intense gaze, neutral dark grey studio backdrop",
            "enabled": True,
            "strength": 1.0,
            "weight_type": "ADDITIVE",
            "start_step": 0.0,
            "end_step": 0.40,
            "merge_mode": "concat"
        },
        "class_type": "SIPromptBlock"
    },
    "6": {
        "inputs": {
            "clip": ["2", 0],
            "label": "Negative",
            "text": "worst quality, blurry, noisy, low resolution, blown out light flares, deformed, extra limbs, bad anatomy, artifacts",
            "enabled": True,
            "strength": 1.0,
            "weight_type": "ADDITIVE",
            "start_step": 0.0,
            "end_step": 1.00,
            "merge_mode": "concat"
        },
        "class_type": "SIPromptBlock"
    },
    "7": {
        "inputs": {
            "preset": "Rembrandt Classic",
            "framing": "Portrait (Classical Bust)",
            "ambient_intensity": 0.05,
            "invert_normals_y": False,
            "key_enabled": True,
            "key_azimuth": 45.0,
            "key_elevation": 35.0,
            "key_intensity": 1.0,
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
            "rim_intensity": 1.10,
            "rim_kelvin": 6500,
            "rim_modifier": "Stripbox",
            "start_step": 0.40,
            "end_step": 1.00,
            "conditioning_strength": 1.00,
            "weight_type": "A_NORM",
            "clip_optional": ["2", 0],
            "conditioning_in": ["5", 0]
        },
        "class_type": "StudioLightRig"
    },
    "8": {"inputs": {"images": ["7", 1]}, "class_type": "PreviewImage"},
    "12": {"inputs": {"images": ["7", 0]}, "class_type": "PreviewImage"},
    "9": {"inputs": {"guidance": 3.5, "conditioning": ["7", 2]}, "class_type": "FluxGuidance"},
    "10": {"inputs": {"width": 896, "height": 1152, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "11": {
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["9", 0],
            "negative": ["6", 0],
            "latent_image": ["10", 0]
        },
        "class_type": "KSampler"
    },
    "13": {"inputs": {"samples": ["11", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
    "14": {"inputs": {"filename_prefix": "StudioLight_T2I", "images": ["13", 0]}, "class_type": "SaveImage"}
}

api_out_path = os.path.join(out_dir, "StudioLight_T2I_Simple_api.json")
with open(api_out_path, "w", encoding="utf-8") as f:
    json.dump(api_prompt, f, indent=2)

print(f"Successfully generated clean T2I API prompt at {api_out_path} with {len(api_prompt)} nodes!")
