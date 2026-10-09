"""
Generates the clean, pure base Text-to-Image workflow for Editorial Fashion Shootings with ComfyUI-StudioLight.
NO LoRAs, NO ChromeCore, NO SI/cybernetic prompts.
Pristine high-fashion portrait photography (Vogue cover quality, natural skin, 85mm lens, 3-point studio lighting).

Zone 1: Pure Base Models (Flux Krea Dev UNET + Dual CLIP + Flux VAE)
Zone 2: Editorial Fashion Prompting & StudioLight Console (Fashion Subject + Suppression Negative + StudioLightRig)
Zone 3: VFX Probes & Deferred Pass (18% Grey Ball + Mirror Chrome Ball + Light Radar + 32-bit Lighting Pass)
Zone 4: Flux Sampling & Vogue Master Output (FluxGuidance 3.5, Latent 896x1152, KSampler, VAEDecode, SaveImage)
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
# 1. PURE BASE MODELS (NO LORAS)
# -------------------------------------------------------------
# Node 1: UNETLoader (Flux Krea Dev)
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
    "title": "Flux Krea Dev (Base UNET)"
}
nodes.append(node_unet)

# Node 2: DualCLIPLoader (T5xxl + CLIP-L)
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

# Node 3: VAELoader (Flux VAE)
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

# -------------------------------------------------------------
# 2. EDITORIAL FASHION PROMPT & STUDIOLIGHT CONSOLE
# -------------------------------------------------------------
# Node 4: SIPromptBlock (High-Fashion Subject: 0.0 -> 0.40)
node_pb_subject = {
    "id": 4,
    "type": "SIPromptBlock",
    "pos": [-800, 100],
    "size": [380, 280],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": [
        {"name": "clip", "type": "CLIP", "link": 2},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None},
        {"name": "mask", "type": "MASK", "link": None}
    ],
    "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [6], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "SIPromptBlock"},
    "widgets_values": [
        "Fashion Subject",
        "cinematic editorial fashion portrait of an elegant woman, high-fashion styling, flawless natural skin texture, subtle freckles, sharp depth of field, neutral dark grey studio backdrop, 85mm portrait lens, 8k resolution, Vogue magazine cover quality",
        True,
        1.0,
        "ADDITIVE",
        0.0,
        0.40,
        "concat"
    ],
    "title": "✨ [Fashion Subject] SIPromptBlock (0.0 → 0.40)"
}
nodes.append(node_pb_subject)
add_link(2, 0, 4, 0, "CLIP")

# Node 5: SIPromptBlock (Universal Suppression Negative: 0.0 -> 1.0)
node_pb_neg = {
    "id": 5,
    "type": "SIPromptBlock",
    "pos": [-800, 420],
    "size": [380, 280],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": [
        {"name": "clip", "type": "CLIP", "link": 3},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None},
        {"name": "mask", "type": "MASK", "link": None}
    ],
    "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [7], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "SIPromptBlock"},
    "widgets_values": [
        "Negative Suppression",
        "blurry, out of focus, low quality, distorted, deformed anatomy, bad eyes, extra limbs, cartoon, 3d render illustration, oversaturated, flat plastic texture, noise, grain, amateur photography",
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
add_link(2, 0, 5, 0, "CLIP")

# Node 6: StudioLightRig (Rembrandt Classic Studio Console)
node_studio_light = {
    "id": 6,
    "type": "StudioLightRig",
    "pos": [-380, 100],
    "size": [380, 720],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": [
        {"name": "normals_optional", "type": "IMAGE", "link": None},
        {"name": "depth_optional", "type": "IMAGE", "link": None},
        {"name": "clip_optional", "type": "CLIP", "link": 4},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": 6},
        {"name": "prompt_text_optional", "type": "STRING", "link": None}
    ],
    "outputs": [
        {"name": "relighting_map", "type": "IMAGE", "links": [8], "shape": 3, "slot_index": 0},
        {"name": "probe_preview", "type": "IMAGE", "links": [9], "shape": 3, "slot_index": 1},
        {"name": "conditioning", "type": "CONDITIONING", "links": [10], "shape": 3, "slot_index": 2},
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
add_link(2, 0, 6, 2, "CLIP")
add_link(4, 0, 6, 3, "CONDITIONING")

# -------------------------------------------------------------
# 3. VFX PROBES & DEFERRED PASS PREVIEWS
# -------------------------------------------------------------
# Node 7: PreviewImage (VFX Probe 18% Grey & Chrome Ball + Radar)
node_probe_prev = {
    "id": 7,
    "type": "PreviewImage",
    "pos": [40, 100],
    "size": [320, 320],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 9}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "✨ VFX Probe (18% Grey & Chrome Ball + Radar)"
}
nodes.append(node_probe_prev)
add_link(6, 1, 7, 0, "IMAGE")

# Node 8: PreviewImage (32-bit Deferred Relighting Pass)
node_relight_prev = {
    "id": 8,
    "type": "PreviewImage",
    "pos": [40, 460],
    "size": [320, 320],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 8}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "✨ 32-bit Deferred Lighting Pass"
}
nodes.append(node_relight_prev)
add_link(6, 0, 8, 0, "IMAGE")

# -------------------------------------------------------------
# 4. FLUX SAMPLING & VOGUE MASTER OUTPUT
# -------------------------------------------------------------
# Node 9: FluxGuidance (3.5)
node_guidance = {
    "id": 9,
    "type": "FluxGuidance",
    "pos": [400, 100],
    "size": [260, 80],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 10}],
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [11], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "FluxGuidance"},
    "widgets_values": [3.5],
    "title": "Flux Guidance (3.5)"
}
nodes.append(node_guidance)
add_link(6, 2, 9, 0, "CONDITIONING")

# Node 10: EmptySD3LatentImage (896x1152 Fashion Editorial Aspect Ratio)
node_latent = {
    "id": 10,
    "type": "EmptySD3LatentImage",
    "pos": [400, 220],
    "size": [260, 110],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [12], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptySD3LatentImage"},
    "widgets_values": [896, 1152, 1],
    "title": "Empty Latent (896x1152)"
}
nodes.append(node_latent)

# Node 11: KSampler (Direct from Base UNET, zero LoRA)
node_ksampler = {
    "id": 11,
    "type": "KSampler",
    "pos": [700, 100],
    "size": [300, 260],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 1},
        {"name": "positive", "type": "CONDITIONING", "link": 11},
        {"name": "negative", "type": "CONDITIONING", "link": 7},
        {"name": "latent_image", "type": "LATENT", "link": 12}
    ],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [13], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [42, "randomize", 25, 1.0, "euler", "beta", 1.0],
    "title": "🚀 Flux KSampler (Pure Base UNET)"
}
nodes.append(node_ksampler)
add_link(1, 0, 11, 0, "MODEL")
add_link(9, 0, 11, 1, "CONDITIONING")
add_link(5, 0, 11, 2, "CONDITIONING")
add_link(10, 0, 11, 3, "LATENT")

# Node 12: VAEDecode
node_decode = {
    "id": 12,
    "type": "VAEDecode",
    "pos": [700, 400],
    "size": [240, 80],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": [
        {"name": "samples", "type": "LATENT", "link": 13},
        {"name": "vae", "type": "VAE", "link": 5}
    ],
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [14], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": [],
    "title": "VAE Decode"
}
nodes.append(node_decode)
add_link(11, 0, 12, 0, "LATENT")
add_link(3, 0, 12, 1, "VAE")

# Node 13: SaveImage (Vogue Master Render)
node_save = {
    "id": 13,
    "type": "SaveImage",
    "pos": [1040, 100],
    "size": [440, 540],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 14}],
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["Fashion_StudioLight"],
    "title": "🎉 [Fashion T2I] Vogue Master Render"
}
nodes.append(node_save)
add_link(12, 0, 13, 0, "IMAGE")

# Assemble final workflow dictionary
workflow_data = {
    "last_node_id": 13,
    "last_link_id": len(links),
    "nodes": nodes,
    "links": links,
    "groups": [
        {"title": "1. PURE BASE MODELS (FLUX KREA DEV — ZERO LORAS)", "bounding": [-1240, 40, 400, 480], "color": "#2a363b"},
        {"title": "2. EDITORIAL FASHION PROMPT & STUDIOLIGHT CONSOLE", "bounding": [-840, 40, 860, 820], "color": "#3f2b47"},
        {"title": "3. VFX PROBES & DEFERRED PASS", "bounding": [0, 40, 380, 780], "color": "#1f3a3d"},
        {"title": "4. FLUX SAMPLING & VOGUE MASTER OUTPUT", "bounding": [360, 40, 1160, 620], "color": "#2e2d4d"}
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.85, "offset": [1300, 80]}},
    "version": 0.4
}

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "StudioLight_T2I_Simple.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(workflow_data, f, indent=2)

print(f"Successfully generated clean Fashion T2I workflow at {out_path} with {len(nodes)} nodes and {len(links)} links!")

# Generate ComfyUI API Prompt format
api_prompt = {
    "1": {"inputs": {"unet_name": "flux1-krea-dev.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "2": {"inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}, "class_type": "DualCLIPLoader"},
    "3": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
    "4": {
        "inputs": {
            "clip": ["2", 0],
            "label": "Fashion Subject",
            "text": "cinematic editorial fashion portrait of an elegant woman, high-fashion styling, flawless natural skin texture, subtle freckles, sharp depth of field, neutral dark grey studio backdrop, 85mm portrait lens, 8k resolution, Vogue magazine cover quality",
            "enabled": True,
            "strength": 1.0,
            "weight_type": "ADDITIVE",
            "start_step": 0.0,
            "end_step": 0.40,
            "merge_mode": "concat"
        },
        "class_type": "SIPromptBlock"
    },
    "5": {
        "inputs": {
            "clip": ["2", 0],
            "label": "Negative Suppression",
            "text": "blurry, out of focus, low quality, distorted, deformed anatomy, bad eyes, extra limbs, cartoon, 3d render illustration, oversaturated, flat plastic texture, noise, grain, amateur photography",
            "enabled": True,
            "strength": 1.0,
            "weight_type": "ADDITIVE",
            "start_step": 0.0,
            "end_step": 1.00,
            "merge_mode": "concat"
        },
        "class_type": "SIPromptBlock"
    },
    "6": {
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
            "conditioning_in": ["4", 0]
        },
        "class_type": "StudioLightRig"
    },
    "7": {"inputs": {"images": ["6", 1]}, "class_type": "PreviewImage"},
    "8": {"inputs": {"images": ["6", 0]}, "class_type": "PreviewImage"},
    "9": {"inputs": {"guidance": 3.5, "conditioning": ["6", 2]}, "class_type": "FluxGuidance"},
    "10": {"inputs": {"width": 896, "height": 1152, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
    "11": {
        "inputs": {
            "seed": 42,
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "beta",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["9", 0],
            "negative": ["5", 0],
            "latent_image": ["10", 0]
        },
        "class_type": "KSampler"
    },
    "12": {"inputs": {"samples": ["11", 0], "vae": ["3", 0]}, "class_type": "VAEDecode"},
    "13": {"inputs": {"filename_prefix": "Fashion_StudioLight", "images": ["12", 0]}, "class_type": "SaveImage"}
}

api_out_path = os.path.join(out_dir, "StudioLight_T2I_Simple_api.json")
with open(api_out_path, "w", encoding="utf-8") as f:
    json.dump(api_prompt, f, indent=2)

print(f"Successfully generated clean Fashion T2I API prompt at {api_out_path} with {len(api_prompt)} nodes!")
