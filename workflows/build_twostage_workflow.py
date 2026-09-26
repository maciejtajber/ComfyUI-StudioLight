#!/usr/bin/env python3
"""
Builds StudioLight_TwoStage_Master.json and StudioLight_TwoStage_Master_api.json
A complete, single-click, two-stage deferred photometric studio lighting workflow for Flux.1.
Stage 1: Generates base unlit/neutral model (16 steps).
Stage 2: Marigold LCM scans real 3D normals (0.4s).
Stage 3: StudioLight calculates 3D physical photons + filmic photometric composite.
Stage 4: Flux refines skin subsurface scattering and catchlights (denoise 0.25, 8 steps).
"""

import json
import os

class WorkflowBuilder:
    def __init__(self):
        self.nodes = {}
        self.links = []
        self.link_counter = 1

    def add_node(self, node_id, node_type, title, pos, size, widgets_values, input_names, output_defs):
        inputs = [{"name": name, "type": type_name, "link": None} for name, type_name in input_names]
        outputs = [{"name": name, "type": type_name, "links": [], "shape": 3, "slot_index": i} for i, (name, type_name) in enumerate(output_defs)]
        node = {
            "id": node_id,
            "type": node_type,
            "pos": pos,
            "size": size,
            "flags": {},
            "order": len(self.nodes) + 1,
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": node_type},
            "widgets_values": widgets_values,
            "title": title
        }
        self.nodes[node_id] = node
        return node

    def connect(self, from_node_id, from_slot, to_node_id, to_slot, type_name):
        l_id = self.link_counter
        self.link_counter += 1
        
        self.links.append([l_id, from_node_id, from_slot, to_node_id, to_slot, type_name])
        
        from_node = self.nodes[from_node_id]
        if from_slot < len(from_node["outputs"]):
            from_node["outputs"][from_slot]["links"].append(l_id)
        else:
            raise ValueError(f"Node {from_node_id} has no output slot {from_slot}")
            
        to_node = self.nodes[to_node_id]
        if to_slot < len(to_node["inputs"]):
            to_node["inputs"][to_slot]["link"] = l_id
        else:
            raise ValueError(f"Node {to_node_id} has no input slot {to_slot}")

    def build_canvas(self, groups=None, offset=None):
        return {
            "last_node_id": max(self.nodes.keys()) if self.nodes else 0,
            "last_link_id": len(self.links),
            "nodes": list(self.nodes.values()),
            "links": self.links,
            "groups": groups or [],
            "config": {},
            "extra": {"ds": {"scale": 0.75, "offset": offset or [1600, 200]}},
            "version": 0.4
        }

wb = WorkflowBuilder()

# --- 1. MODEL STACK ---
wb.add_node(
    node_id=1,
    node_type="UNETLoader",
    title="Flux Krea Dev (Base UNET)",
    pos=[-1600, 100],
    size=[320, 90],
    widgets_values=["flux1-krea-dev.safetensors", "default"],
    input_names=[],
    output_defs=[("MODEL", "MODEL")]
)

wb.add_node(
    node_id=2,
    node_type="DualCLIPLoader",
    title="Dual CLIP Loader (T5 + CLIP-L)",
    pos=[-1600, 230],
    size=[320, 110],
    widgets_values=["t5xxl_fp16.safetensors", "clip_l.safetensors", "flux"],
    input_names=[],
    output_defs=[("CLIP", "CLIP")]
)

wb.add_node(
    node_id=3,
    node_type="VAELoader",
    title="Flux VAE",
    pos=[-1600, 380],
    size=[320, 80],
    widgets_values=["ae.safetensors"],
    input_names=[],
    output_defs=[("VAE", "VAE")]
)

# --- 2. PROMPT BLOCKS ---
wb.add_node(
    node_id=4,
    node_type="SIPromptBlock",
    title="✨ [Fashion Subject] SIPromptBlock (0.0 → 0.40)",
    pos=[-1220, 100],
    size=[380, 200],
    widgets_values=[
        4,
        "Vogue editorial high-fashion beauty portrait of an elegant woman, exquisite bone structure, flawless natural skin texture, haute couture styling, neutral studio backdrop, 85mm lens, f/2.8, hyperrealistic photography",
        0.0,
        0.40,
        1.0,
        "A_NORM"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

wb.add_node(
    node_id=5,
    node_type="SIPromptBlock",
    title="⛔ [Negative] Universal Suppression",
    pos=[-1220, 340],
    size=[380, 160],
    widgets_values=[
        5,
        "plastic skin, mannequin, low quality, oversaturated, blurry, bad anatomy, deformed, watermark, signature",
        0.0,
        1.0,
        1.0,
        "A_NORM"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

# --- 3. STAGE 1: DRAFT GENERATION ---
wb.add_node(
    node_id=6,
    node_type="EmptySD3LatentImage",
    title="Empty Latent (896x1152)",
    pos=[-1220, 540],
    size=[320, 110],
    widgets_values=[896, 1152, 1],
    input_names=[],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=7,
    node_type="FluxGuidance",
    title="Flux Guidance Stage 1 (3.5)",
    pos=[-800, 100],
    size=[240, 60],
    widgets_values=[3.5],
    input_names=[("conditioning", "CONDITIONING")],
    output_defs=[("CONDITIONING", "CONDITIONING")]
)

wb.add_node(
    node_id=8,
    node_type="KSampler",
    title="🚀 Stage 1: Subject Genesis (16 Steps)",
    pos=[-800, 200],
    size=[320, 290],
    widgets_values=[42, "fixed", 16, 1.0, "euler", "beta", 1.0],
    input_names=[
        ("model", "MODEL"),
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent_image", "LATENT")
    ],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=9,
    node_type="VAEDecode",
    title="VAE Decode (Stage 1 Draft)",
    pos=[-440, 200],
    size=[220, 50],
    widgets_values=[],
    input_names=[("samples", "LATENT"), ("vae", "VAE")],
    output_defs=[("IMAGE", "IMAGE")]
)

wb.add_node(
    node_id=10,
    node_type="PreviewImage",
    title="✨ [Stage 1] Subject Draft Preview",
    pos=[-440, 290],
    size=[320, 380],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# --- 4. 3D NORMAL EXTRACTION (MARIGOLD LCM) ---
wb.add_node(
    node_id=11,
    node_type="MarigoldModelLoader",
    title="Marigold Normals LCM Loader",
    pos=[-80, 100],
    size=[320, 80],
    widgets_values=["prs-eth/marigold-normals-lcm-v0-1"],
    input_names=[],
    output_defs=[("marigold_model", "MARIGOLD_MODEL")]
)

wb.add_node(
    node_id=12,
    node_type="MarigoldDepthEstimation_v2",
    title="✨ [Instant 3D Scanner] Marigold Normals (4 Steps)",
    pos=[-80, 220],
    size=[320, 220],
    widgets_values=[123, "fixed", 4, 1, 896, "LCMScheduler", False, True],
    input_names=[("marigold_model", "MARIGOLD_MODEL"), ("image", "IMAGE")],
    output_defs=[("image", "IMAGE")]
)

wb.add_node(
    node_id=13,
    node_type="PreviewImage",
    title="✨ [Extracted] 3D Surface Normals",
    pos=[-80, 480],
    size=[320, 380],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# --- 5. STUDIOLIGHT PHOTOMETRIC ENGINE ---
wb.add_node(
    node_id=14,
    node_type="StudioLightRig",
    title="🎛️ StudioLight (Master Photographic Console)",
    pos=[300, 100],
    size=[420, 720],
    widgets_values=[
        "Rembrandt Classic",           # preset
        "Portrait (Classical Bust)",   # framing
        0.05,                          # ambient_intensity
        False,                         # invert_normals_y
        True,                          # key_enabled
        45.0,                          # key_azimuth
        35.0,                          # key_elevation
        1.0,                           # key_intensity
        5500,                          # key_kelvin
        "Octabox (Soft)",              # key_modifier
        True,                          # fill_enabled
        -30.0,                         # fill_azimuth
        15.0,                          # fill_elevation
        0.35,                          # fill_intensity
        4800,                          # fill_kelvin
        "Octabox (Soft)",              # fill_modifier
        True,                          # rim_enabled
        160.0,                         # rim_azimuth
        45.0,                          # rim_elevation
        1.10,                          # rim_intensity
        6500,                          # rim_kelvin
        "Stripbox",                    # rim_modifier
        0.40,                          # start_step
        1.00,                          # end_step
        1.00,                          # conditioning_strength
        "A_NORM"                       # weight_type
    ],
    input_names=[
        ("normals_optional", "IMAGE"),
        ("depth_optional", "IMAGE"),
        ("image_optional", "IMAGE"),
        ("clip_optional", "CLIP"),
        ("conditioning_in", "CONDITIONING"),
        ("prompt_text_optional", "STRING")
    ],
    output_defs=[
        ("relighting_map", "IMAGE"),
        ("probe_preview", "IMAGE"),
        ("conditioning", "CONDITIONING"),
        ("lighting_prompt", "STRING"),
        ("lit_composite", "IMAGE")
    ]
)

wb.add_node(
    node_id=15,
    node_type="PreviewImage",
    title="✨ VFX Probe (18% Grey & Chrome Ball + Radar)",
    pos=[760, 100],
    size=[320, 320],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

wb.add_node(
    node_id=16,
    node_type="PreviewImage",
    title="✨ 32-bit Photometric Lit Composite",
    pos=[760, 460],
    size=[320, 380],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# --- 6. STAGE 2: PHOTOMETRIC POLISH ---
wb.add_node(
    node_id=17,
    node_type="VAEEncode",
    title="VAE Encode (Lit Composite)",
    pos=[1120, 100],
    size=[240, 50],
    widgets_values=[],
    input_names=[("pixels", "IMAGE"), ("vae", "VAE")],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=18,
    node_type="FluxGuidance",
    title="Flux Guidance Stage 2 (3.5)",
    pos=[1120, 190],
    size=[240, 60],
    widgets_values=[3.5],
    input_names=[("conditioning", "CONDITIONING")],
    output_defs=[("CONDITIONING", "CONDITIONING")]
)

wb.add_node(
    node_id=19,
    node_type="KSampler",
    title="💎 Stage 2: Photometric Polish (Denoise 0.25)",
    pos=[1120, 290],
    size=[320, 290],
    widgets_values=[1042, "fixed", 24, 1.0, "euler", "beta", 0.25],
    input_names=[
        ("model", "MODEL"),
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent_image", "LATENT")
    ],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=20,
    node_type="VAEDecode",
    title="VAE Decode (Final Master)",
    pos=[1480, 290],
    size=[220, 50],
    widgets_values=[],
    input_names=[("samples", "LATENT"), ("vae", "VAE")],
    output_defs=[("IMAGE", "IMAGE")]
)

wb.add_node(
    node_id=21,
    node_type="SaveImage",
    title="🎉 [Vogue Master] StudioLight Final Render",
    pos=[1480, 380],
    size=[420, 500],
    widgets_values=["StudioLight_Master"],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# --- 7. CONNECTIONS (WIRES) ---

# Dual CLIP to prompt blocks & StudioLight
wb.connect(2, 0, 4, 0, "CLIP")       # CLIP -> Subject PromptBlock
wb.connect(2, 0, 5, 0, "CLIP")       # CLIP -> Negative PromptBlock
wb.connect(2, 0, 14, 3, "CLIP")      # CLIP -> StudioLightRig.clip_optional

# Upstream prompt chain to StudioLight
wb.connect(4, 0, 14, 4, "CONDITIONING") # Subject conditioning -> StudioLight.conditioning_in

# Stage 1 Guidance & Sampling
wb.connect(4, 0, 7, 0, "CONDITIONING")  # Subject conditioning -> FluxGuidance 1
wb.connect(1, 0, 8, 0, "MODEL")         # Flux MODEL -> KSampler 1
wb.connect(7, 0, 8, 1, "CONDITIONING")  # Guidance 1 -> KSampler 1 pos
wb.connect(5, 0, 8, 2, "CONDITIONING")  # Negative -> KSampler 1 neg
wb.connect(6, 0, 8, 3, "LATENT")        # Empty Latent -> KSampler 1 latent

# Stage 1 Decode & Normal Extraction
wb.connect(8, 0, 9, 0, "LATENT")        # KSampler 1 -> VAE Decode 1
wb.connect(3, 0, 9, 1, "VAE")           # VAE -> VAE Decode 1
wb.connect(9, 0, 10, 0, "IMAGE")        # VAE Decode 1 -> Draft Preview
wb.connect(11, 0, 12, 0, "MARIGOLD_MODEL") # Marigold Model -> Estimator
wb.connect(9, 0, 12, 1, "IMAGE")        # Draft Image -> Marigold Estimator
wb.connect(12, 0, 13, 0, "IMAGE")       # Marigold Normals -> Normals Preview

# StudioLight inputs
wb.connect(12, 0, 14, 0, "IMAGE")       # Marigold Normals -> StudioLight.normals_optional
wb.connect(9, 0, 14, 2, "IMAGE")        # Draft Image -> StudioLight.image_optional

# StudioLight outputs & Previews
wb.connect(14, 1, 15, 0, "IMAGE")       # probe_preview -> VFX Probe Preview
wb.connect(14, 4, 16, 0, "IMAGE")       # lit_composite -> Lit Composite Preview

# Stage 2 Encode & Refinement
wb.connect(14, 4, 17, 0, "IMAGE")       # lit_composite -> VAE Encode 2
wb.connect(3, 0, 17, 1, "VAE")          # VAE -> VAE Encode 2
wb.connect(14, 2, 18, 0, "CONDITIONING")# StudioLight conditioning -> FluxGuidance 2
wb.connect(1, 0, 19, 0, "MODEL")        # Flux MODEL -> KSampler 2
wb.connect(18, 0, 19, 1, "CONDITIONING")# Guidance 2 -> KSampler 2 pos
wb.connect(5, 0, 19, 2, "CONDITIONING") # Negative -> KSampler 2 neg
wb.connect(17, 0, 19, 3, "LATENT")      # VAE Encode 2 -> KSampler 2 latent

# Final Decode & Save
wb.connect(19, 0, 20, 0, "LATENT")      # KSampler 2 -> VAE Decode 2
wb.connect(3, 0, 20, 1, "VAE")          # VAE -> VAE Decode 2
wb.connect(20, 0, 21, 0, "IMAGE")       # Final Image -> SaveImage

# --- BUILD JSON FILES ---
out_dir = os.path.dirname(os.path.abspath(__file__))

canvas_json = wb.build_canvas(
    groups=[
        {"title": "1. Base Models & Subject Casting", "bounding": [-1640, 40, 1180, 680], "color": "#2a363b"},
        {"title": "2. Instant 3D Normal Extraction (Marigold LCM)", "bounding": [-120, 40, 380, 840], "color": "#1e3d59"},
        {"title": "3. 3-Point Studio Lighting Console (Physics)", "bounding": [280, 40, 820, 840], "color": "#17252a"},
        {"title": "4. Photometric Skin Polish & Master Render", "bounding": [1100, 40, 840, 860], "color": "#3f2b96"}
    ]
)

canvas_path = os.path.join(out_dir, "StudioLight_TwoStage_Master.json")
with open(canvas_path, "w", encoding="utf-8") as f:
    json.dump(canvas_json, f, indent=2)
print(f"✓ Canvas workflow generated: {canvas_path} ({len(canvas_json['nodes'])} nodes, {len(canvas_json['links'])} links)")

# API Format dictionary
api_json = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-krea-dev.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5xxl_fp16.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
    "4": {"class_type": "SIPromptBlock", "inputs": {"clip": ["2", 0], "block_index": 4, "text": "Vogue editorial high-fashion beauty portrait of an elegant woman, exquisite bone structure, flawless natural skin texture, haute couture styling, neutral studio backdrop, 85mm lens, f/2.8, hyperrealistic photography", "start_step": 0.0, "end_step": 0.40, "strength": 1.0, "weight_type": "A_NORM"}},
    "5": {"class_type": "SIPromptBlock", "inputs": {"clip": ["2", 0], "block_index": 5, "text": "plastic skin, mannequin, low quality, oversaturated, blurry, bad anatomy, deformed, watermark, signature", "start_step": 0.0, "end_step": 1.0, "strength": 1.0, "weight_type": "A_NORM"}},
    "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 896, "height": 1152, "batch_size": 1}},
    "7": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 3.5}},
    "8": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": 42, "control_after_generate": "fixed", "steps": 16, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
    "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
    "10": {"class_type": "PreviewImage", "inputs": {"images": ["9", 0]}},
    "11": {"class_type": "MarigoldModelLoader", "inputs": {"model": "prs-eth/marigold-normals-lcm-v0-1"}},
    "12": {"class_type": "MarigoldDepthEstimation_v2", "inputs": {"marigold_model": ["11", 0], "image": ["9", 0], "seed": 123, "denoise_steps": 4, "n_repeat": 1, "processing_res": 896, "scheduler": "LCMScheduler", "color_map": False, "invert": True}},
    "13": {"class_type": "PreviewImage", "inputs": {"images": ["12", 0]}},
    "14": {
        "class_type": "StudioLightRig",
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
            "normals_optional": ["12", 0],
            "image_optional": ["9", 0],
            "clip_optional": ["2", 0],
            "conditioning_in": ["4", 0]
        }
    },
    "15": {"class_type": "PreviewImage", "inputs": {"images": ["14", 1]}},
    "16": {"class_type": "PreviewImage", "inputs": {"images": ["14", 4]}},
    "17": {"class_type": "VAEEncode", "inputs": {"pixels": ["14", 4], "vae": ["3", 0]}},
    "18": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["14", 2], "guidance": 3.5}},
    "19": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["18", 0], "negative": ["5", 0], "latent_image": ["17", 0], "seed": 1042, "control_after_generate": "fixed", "steps": 24, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta", "denoise": 0.25}},
    "20": {"class_type": "VAEDecode", "inputs": {"samples": ["19", 0], "vae": ["3", 0]}},
    "21": {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": "StudioLight_Master"}}
}

api_path = os.path.join(out_dir, "StudioLight_TwoStage_Master_api.json")
with open(api_path, "w", encoding="utf-8") as f:
    json.dump(api_json, f, indent=2)
print(f"✓ API workflow generated: {api_path} ({len(api_json)} nodes)")
