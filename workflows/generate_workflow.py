"""
Generates the Master Unified Workflow JSON for ComfyUI-StudioLight:
Zone 1: Models & LoRAs (Flux Krea Dev, Union Pro 2, ChromeCore)
Zone 2: Text-to-Image Studio Generation (PromptBlock + StudioLightRig + KSampler)
Zone 3: Image-to-Image Relighting (Opalin Portrait + Normal Preprocessor + StudioLightRig + ControlNet Relight)
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
# 1. MODELS & LORAS
# -------------------------------------------------------------
# Node 1: UNETLoader (Flux Krea Dev)
node_unet = {
    "id": 1,
    "type": "UNETLoader",
    "pos": [-1400, 100],
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

# Node 2: DualCLIPLoader (T5xxl + CLIP-L)
node_clip = {
    "id": 2,
    "type": "DualCLIPLoader",
    "pos": [-1400, 240],
    "size": [320, 110],
    "flags": {},
    "order": 2,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2, 3], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "DualCLIPLoader"},
    "widgets_values": ["t5xxl_fp16.safetensors", "clip_l.safetensors", "flux"],
    "title": "Dual CLIP Loader (T5 + CLIP-L)"
}
nodes.append(node_clip)

# Node 3: VAELoader (ae.safetensors)
node_vae = {
    "id": 3,
    "type": "VAELoader",
    "pos": [-1400, 400],
    "size": [320, 80],
    "flags": {},
    "order": 3,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "VAE", "type": "VAE", "links": [4, 5, 6], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAELoader"},
    "widgets_values": ["ae.safetensors"],
    "title": "Flux VAE"
}
nodes.append(node_vae)

# Node 4: LoRALoaderModelOnly (chromecore_flux)
node_lora = {
    "id": 4,
    "type": "LoraLoaderModelOnly",
    "pos": [-1000, 100],
    "size": [320, 90],
    "flags": {},
    "order": 4,
    "mode": 0,
    "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
    "outputs": [{"name": "MODEL", "type": "MODEL", "links": [7, 8], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "LoraLoaderModelOnly"},
    "widgets_values": ["chromecore_flux.safetensors", 0.75],
    "title": "ChromeCore LoRA"
}
nodes.append(node_lora)
add_link(1, 0, 4, 0, "MODEL")

# Node 5: ControlNetLoader (flux-union-pro-2)
node_cn_loader = {
    "id": 5,
    "type": "ControlNetLoader",
    "pos": [-1000, 240],
    "size": [320, 80],
    "flags": {},
    "order": 5,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "CONTROL_NET", "type": "CONTROL_NET", "links": [9], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "ControlNetLoader"},
    "widgets_values": ["flux-union-pro-2.safetensors"],
    "title": "ControlNet Union Pro 2"
}
nodes.append(node_cn_loader)

# -------------------------------------------------------------
# 2. LANE A: TEXT-TO-IMAGE STUDIO GENERATION
# -------------------------------------------------------------
# Node 10: PromptBlock (Subject)
node_pb1 = {
    "id": 10,
    "type": "PromptBlock",
    "pos": [-600, 100],
    "size": [380, 280],
    "flags": {},
    "order": 6,
    "mode": 0,
    "inputs": [
        {"name": "clip", "type": "CLIP", "link": 2},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None},
        {"name": "mask", "type": "MASK", "link": None}
    ],
    "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [10], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "PromptBlock"},
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
    "title": "[T2I] Subject PromptBlock (0.0 → 0.40)"
}
nodes.append(node_pb1)
add_link(2, 0, 10, 0, "CLIP")

# Node 11: StudioLightRig (T2I Lighting Console)
node_sl1 = {
    "id": 11,
    "type": "StudioLightRig",
    "pos": [-180, 100],
    "size": [380, 720],
    "flags": {},
    "order": 7,
    "mode": 0,
    "inputs": [
        {"name": "normals_optional", "type": "IMAGE", "link": None},
        {"name": "depth_optional", "type": "IMAGE", "link": None},
        {"name": "clip_optional", "type": "CLIP", "link": None},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": 10}
    ],
    "outputs": [
        {"name": "relighting_map", "type": "IMAGE", "links": [], "shape": 3, "slot_index": 0},
        {"name": "probe_preview", "type": "IMAGE", "links": [11], "shape": 3, "slot_index": 1},
        {"name": "conditioning", "type": "CONDITIONING", "links": [12], "shape": 3, "slot_index": 2},
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
    "title": "[T2I] StudioLight (Rembrandt Console)"
}
nodes.append(node_sl1)
add_link(10, 0, 11, 3, "CONDITIONING")

# Node 12: PreviewImage (VFX Probe Preview T2I)
node_prev1 = {
    "id": 12,
    "type": "PreviewImage",
    "pos": [240, 100],
    "size": [320, 320],
    "flags": {},
    "order": 8,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 11}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "[T2I] VFX Probe & Studio Radar"
}
nodes.append(node_prev1)
add_link(11, 1, 12, 0, "IMAGE")

# Node 13: EmptySD3LatentImage
node_latent1 = {
    "id": 13,
    "type": "EmptySD3LatentImage",
    "pos": [240, 460],
    "size": [280, 110],
    "flags": {},
    "order": 9,
    "mode": 0,
    "inputs": [],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [13], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "EmptySD3LatentImage"},
    "widgets_values": [896, 1152, 1],
    "title": "Empty Latent (896x1152)"
}
nodes.append(node_latent1)

# Node 14: FluxGuidance (T2I)
node_guidance1 = {
    "id": 14,
    "type": "FluxGuidance",
    "pos": [240, 600],
    "size": [280, 80],
    "flags": {},
    "order": 10,
    "mode": 0,
    "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 12}],
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [14], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "FluxGuidance"},
    "widgets_values": [3.5],
    "title": "Flux Guidance (3.5)"
}
nodes.append(node_guidance1)
add_link(11, 2, 14, 0, "CONDITIONING")

# Node 15: KSampler (T2I)
node_ksampler1 = {
    "id": 15,
    "type": "KSampler",
    "pos": [560, 460],
    "size": [300, 260],
    "flags": {},
    "order": 11,
    "mode": 0,
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 7},
        {"name": "positive", "type": "CONDITIONING", "link": 14},
        {"name": "negative", "type": "CONDITIONING", "link": None},
        {"name": "latent_image", "type": "LATENT", "link": 13}
    ],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [15], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [42, "randomize", 25, 1.0, "euler", "beta", 1.0],
    "title": "[T2I] Flux KSampler"
}
nodes.append(node_ksampler1)
add_link(4, 0, 15, 0, "MODEL")
add_link(14, 0, 15, 1, "CONDITIONING")
add_link(13, 0, 15, 3, "LATENT")

# Node 16: VAEDecode (T2I)
node_decode1 = {
    "id": 16,
    "type": "VAEDecode",
    "pos": [900, 460],
    "size": [220, 80],
    "flags": {},
    "order": 12,
    "mode": 0,
    "inputs": [
        {"name": "samples", "type": "LATENT", "link": 15},
        {"name": "vae", "type": "VAE", "link": 4}
    ],
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [16], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": [],
    "title": "VAE Decode"
}
nodes.append(node_decode1)
add_link(15, 0, 16, 0, "LATENT")
add_link(3, 0, 16, 1, "VAE")

# Node 17: SaveImage (T2I Result)
node_save1 = {
    "id": 17,
    "type": "SaveImage",
    "pos": [1160, 460],
    "size": [400, 480],
    "flags": {},
    "order": 13,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 16}],
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["StudioLight_T2I"],
    "title": "🎉 [T2I] Master Render"
}
nodes.append(node_save1)
add_link(16, 0, 17, 0, "IMAGE")

# -------------------------------------------------------------
# 3. LANE B: IMAGE-TO-IMAGE & OPALIN RELIGHTING
# -------------------------------------------------------------
# Node 20: LoadImage (Opalin Portrait)
node_load = {
    "id": 20,
    "type": "LoadImage",
    "pos": [-600, 900],
    "size": [320, 320],
    "flags": {},
    "order": 14,
    "mode": 0,
    "inputs": [],
    "outputs": [
        {"name": "IMAGE", "type": "IMAGE", "links": [17, 18], "shape": 3, "slot_index": 0},
        {"name": "MASK", "type": "MASK", "links": [], "shape": 3, "slot_index": 1}
    ],
    "properties": {"Node name for S&R": "LoadImage"},
    "widgets_values": ["opalin_portrait.jpg", "image"],
    "title": "[I2I] Load Opalin Portrait"
}
nodes.append(node_load)

# Node 21: StudioLightRig (I2I Relighting Console)
node_sl2 = {
    "id": 21,
    "type": "StudioLightRig",
    "pos": [-180, 900],
    "size": [380, 720],
    "flags": {},
    "order": 15,
    "mode": 0,
    "inputs": [
        {"name": "normals_optional", "type": "IMAGE", "link": None},
        {"name": "depth_optional", "type": "IMAGE", "link": None},
        {"name": "clip_optional", "type": "CLIP", "link": 3},
        {"name": "conditioning_in", "type": "CONDITIONING", "link": None}
    ],
    "outputs": [
        {"name": "relighting_map", "type": "IMAGE", "links": [19, 20], "shape": 3, "slot_index": 0},
        {"name": "probe_preview", "type": "IMAGE", "links": [21], "shape": 3, "slot_index": 1},
        {"name": "conditioning", "type": "CONDITIONING", "links": [22], "shape": 3, "slot_index": 2},
        {"name": "lighting_prompt", "type": "STRING", "links": [], "shape": 3, "slot_index": 3}
    ],
    "properties": {"Node name for S&R": "StudioLightRig"},
    "widgets_values": [
        "Cinematic Film Noir",
        "Portrait (Classical Bust)",
        0.02,
        False,
        True, 75.0, 20.0, 1.5, 3200, "Fresnel Spotlight",
        False, -45.0, 0.0, 0.05, 3200, "Octabox (Soft)",
        True, -150.0, 35.0, 1.4, 6500, "Stripbox",
        0.35, 1.00, 1.00, "A_NORM"
    ],
    "title": "[I2I] StudioLight (Film Noir Relighting)"
}
nodes.append(node_sl2)
add_link(2, 0, 21, 2, "CLIP")

# Node 22: PreviewImage (Relighting Map Preview)
node_prev_map = {
    "id": 22,
    "type": "PreviewImage",
    "pos": [240, 900],
    "size": [300, 300],
    "flags": {},
    "order": 16,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 19}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "[I2I] Calculated Relighting Map (32-bit)"
}
nodes.append(node_prev_map)
add_link(21, 0, 22, 0, "IMAGE")

# Node 23: PreviewImage (VFX Probe Preview I2I)
node_prev_probe2 = {
    "id": 23,
    "type": "PreviewImage",
    "pos": [240, 1240],
    "size": [300, 300],
    "flags": {},
    "order": 17,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 21}],
    "outputs": [],
    "properties": {"Node name for S&R": "PreviewImage"},
    "widgets_values": [],
    "title": "[I2I] VFX Probe & Studio Radar"
}
nodes.append(node_prev_probe2)
add_link(21, 1, 23, 0, "IMAGE")

# Node 24: FluxGuidance (I2I)
node_guidance2 = {
    "id": 24,
    "type": "FluxGuidance",
    "pos": [580, 900],
    "size": [260, 80],
    "flags": {},
    "order": 18,
    "mode": 0,
    "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 22}],
    "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [23], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "FluxGuidance"},
    "widgets_values": [3.5],
    "title": "Flux Guidance (3.5)"
}
nodes.append(node_guidance2)
add_link(21, 2, 24, 0, "CONDITIONING")

# Node 25: VAEEncode (Opalin to Latents)
node_encode2 = {
    "id": 25,
    "type": "VAEEncode",
    "pos": [580, 1020],
    "size": [240, 80],
    "flags": {},
    "order": 19,
    "mode": 0,
    "inputs": [
        {"name": "pixels", "type": "IMAGE", "link": 18},
        {"name": "vae", "type": "VAE", "link": 5}
    ],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [24], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEEncode"},
    "widgets_values": [],
    "title": "VAE Encode (Opalin)"
}
nodes.append(node_encode2)
add_link(20, 0, 25, 0, "IMAGE")
add_link(3, 0, 25, 1, "VAE")

# Node 26: KSampler (I2I Relighting Pass)
node_ksampler2 = {
    "id": 26,
    "type": "KSampler",
    "pos": [860, 900],
    "size": [300, 260],
    "flags": {},
    "order": 20,
    "mode": 0,
    "inputs": [
        {"name": "model", "type": "MODEL", "link": 8},
        {"name": "positive", "type": "CONDITIONING", "link": 23},
        {"name": "negative", "type": "CONDITIONING", "link": None},
        {"name": "latent_image", "type": "LATENT", "link": 24}
    ],
    "outputs": [{"name": "LATENT", "type": "LATENT", "links": [25], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "KSampler"},
    "widgets_values": [1997, "randomize", 25, 1.0, "euler", "beta", 0.65],
    "title": "[I2I] Flux Relighting KSampler (Denoise 0.65)"
}
nodes.append(node_ksampler2)
add_link(4, 0, 26, 0, "MODEL")
add_link(24, 0, 26, 1, "CONDITIONING")
add_link(25, 0, 26, 3, "LATENT")

# Node 27: VAEDecode (I2I)
node_decode2 = {
    "id": 27,
    "type": "VAEDecode",
    "pos": [1180, 900],
    "size": [220, 80],
    "flags": {},
    "order": 21,
    "mode": 0,
    "inputs": [
        {"name": "samples", "type": "LATENT", "link": 25},
        {"name": "vae", "type": "VAE", "link": 6}
    ],
    "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [26], "shape": 3, "slot_index": 0}],
    "properties": {"Node name for S&R": "VAEDecode"},
    "widgets_values": [],
    "title": "VAE Decode"
}
nodes.append(node_decode2)
add_link(26, 0, 27, 0, "LATENT")
add_link(3, 0, 27, 1, "VAE")

# Node 28: SaveImage (I2I Relit Result)
node_save2 = {
    "id": 28,
    "type": "SaveImage",
    "pos": [1440, 900],
    "size": [420, 500],
    "flags": {},
    "order": 22,
    "mode": 0,
    "inputs": [{"name": "images", "type": "IMAGE", "link": 26}],
    "outputs": [],
    "properties": {"Node name for S&R": "SaveImage"},
    "widgets_values": ["StudioLight_Opalin_Relit"],
    "title": "🎉 [I2I] Opalin Relit Master"
}
nodes.append(node_save2)
add_link(27, 0, 28, 0, "IMAGE")

# Assemble final workflow dictionary
workflow_data = {
    "last_node_id": 28,
    "last_link_id": len(links),
    "nodes": nodes,
    "links": links,
    "groups": [
        {"title": "1. MODELS & LORAS (FLUX KREA DEV + CHROMECORE)", "bounding": [-1440, 40, 800, 480], "color": "#2a363b"},
        {"title": "2. LANE A: TEXT-TO-IMAGE STUDIO LIGHTING", "bounding": [-640, 40, 2240, 800], "color": "#3f2b47"},
        {"title": "3. LANE B: IMAGE-TO-IMAGE & OPALIN RELIGHTING (DENOISE 0.65)", "bounding": [-640, 840, 2540, 800], "color": "#1f3a3d"}
    ],
    "config": {},
    "extra": {"ds": {"scale": 0.8, "offset": [1500, 100]}},
    "version": 0.4
}

out_path = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_Master_Suite.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(workflow_data, f, indent=2)

print(f"Successfully generated Master Suite workflow at {out_path} with {len(nodes)} nodes and {len(links)} links!")
