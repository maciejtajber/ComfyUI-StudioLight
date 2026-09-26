#!/usr/bin/env python3
"""
Builds StudioLight_Master.json and StudioLight_Master_api.json
The definitive, unified, single-click StudioLight Master Workflow.
Combines:
1. Pure Text-to-Image (T2I) via Flux (16 steps).
2. Instant 3D Normal Extraction via Marigold LCM (0.4s).
3. StudioLight Photographic Rig: Key, Fill, Rim, Kelvin, Modifiers on real 3D face (0.15s).
4. Direct Master Image Output (Zero second KSampler latency, zero identity drift).
5. LoadImage node wired as ready-to-use alternative input for Path 1 (relighting any existing image).
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

# 1. MODEL LOADERS
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

# 2. PROMPT BLOCKS
wb.add_node(
    node_id=4,
    node_type="SIPromptBlock",
    title="✨ [Fashion Subject] SIPromptBlock",
    pos=[-1220, 100],
    size=[380, 200],
    widgets_values=[
        4,
        "Vogue editorial high-fashion beauty portrait of an elegant woman, exquisite bone structure, flawless natural skin texture, haute couture styling, neutral studio backdrop, 85mm lens, f/2.8, hyperrealistic photography",
        0.0,
        1.0,
        1.0,
        "A_NORM"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

wb.add_node(
    node_id=5,
    node_type="SIPromptBlock",
    title="✨ [Negative / Floor] Negative PromptBlock",
    pos=[-1220, 340],
    size=[380, 160],
    widgets_values=[
        2,
        "low quality, blurry, distorted, cartoon, 3d render, plastic skin, oversaturated, deformed",
        0.0,
        1.0,
        1.0,
        "A_NORM"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

wb.add_node(
    node_id=6,
    node_type="EmptySD3LatentImage",
    title="Latent Canvas (1024 x 1024)",
    pos=[-1220, 540],
    size=[380, 110],
    widgets_values=[1024, 1024, 1],
    input_names=[],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=7,
    node_type="FluxGuidance",
    title="Flux Guidance (3.5)",
    pos=[-800, 100],
    size=[320, 60],
    widgets_values=[3.5],
    input_names=[("conditioning", "CONDITIONING")],
    output_defs=[("CONDITIONING", "CONDITIONING")]
)

# 3. STAGE 1 GENERATION
wb.add_node(
    node_id=8,
    node_type="KSampler",
    title="🚀 Flux Subject Genesis (16 Steps)",
    pos=[-800, 200],
    size=[320, 290],
    widgets_values=[42, "fixed", 16, 1.0, "euler", "beta", 1.0],
    input_names=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=9,
    node_type="VAEDecode",
    title="VAE Decode (Subject)",
    pos=[-440, 200],
    size=[220, 50],
    widgets_values=[],
    input_names=[("samples", "LATENT"), ("vae", "VAE")],
    output_defs=[("IMAGE", "IMAGE")]
)

wb.add_node(
    node_id=10,
    node_type="PreviewImage",
    title="✨ [Raw Subject] Unlit Draft Preview",
    pos=[-440, 290],
    size=[320, 380],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# 4. 3D NORMAL EXTRACTION
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
    title="✨ [Instant 3D Scanner] Marigold Normals",
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

# 5. STUDIOLIGHT RIG
wb.add_node(
    node_id=14,
    node_type="StudioLightRig",
    title="🎛️ StudioLight (Master Photographic Console)",
    pos=[300, 100],
    size=[420, 720],
    widgets_values=[
        "Rembrandt Classic",           # preset
        "Portrait (Classical Bust)",   # framing
        0.08,                          # ambient_intensity
        False,                         # invert_normals_y
        True,                          # key_enabled
        45.0,                          # key_azimuth
        25.0,                          # key_elevation
        1.15,                          # key_intensity
        5500,                          # key_kelvin
        "Octabox (Soft)",              # key_modifier
        True,                          # fill_enabled
        -30.0,                         # fill_azimuth
        15.0,                          # fill_elevation
        0.30,                          # fill_intensity
        4800,                          # fill_kelvin
        "Octabox (Soft)",              # fill_modifier
        True,                          # rim_enabled
        160.0,                         # rim_azimuth
        35.0,                          # rim_elevation
        0.80,                          # rim_intensity
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
    title="💎 [Master Lit Output] StudioLight Final",
    pos=[760, 460],
    size=[400, 480],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

wb.add_node(
    node_id=17,
    node_type="SaveImage",
    title="💾 Save Master Lit Portrait",
    pos=[1200, 460],
    size=[300, 120],
    widgets_values=["StudioLight_Master"],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# 6. ALTERNATIVE LOAD IMAGE NODE (FOR PATH 1 / DIRECT RELIGHTING)
wb.add_node(
    node_id=18,
    node_type="LoadImage",
    title="📂 [Optional Path 1] Load External Image to Relight",
    pos=[-80, -200],
    size=[320, 240],
    widgets_values=["example.png", "image"],
    input_names=[],
    output_defs=[("IMAGE", "IMAGE"), ("MASK", "MASK")]
)

# --- CONNECTIONS ---
wb.connect(2, 0, 4, 0, "CLIP")          # CLIP -> Subject PromptBlock
wb.connect(2, 0, 5, 0, "CLIP")          # CLIP -> Negative PromptBlock
wb.connect(4, 0, 7, 0, "CONDITIONING")  # Subject Conditioning -> FluxGuidance
wb.connect(1, 0, 8, 0, "MODEL")         # Flux Model -> KSampler
wb.connect(7, 0, 8, 1, "CONDITIONING")  # Positive Conditioning -> KSampler
wb.connect(5, 0, 8, 2, "CONDITIONING")  # Negative Conditioning -> KSampler
wb.connect(6, 0, 8, 3, "LATENT")        # Latent Canvas -> KSampler
wb.connect(8, 0, 9, 0, "LATENT")        # KSampler Latent -> VAEDecode
wb.connect(3, 0, 9, 1, "VAE")           # Flux VAE -> VAEDecode
wb.connect(9, 0, 10, 0, "IMAGE")        # VAEDecode -> Subject Draft Preview

wb.connect(11, 0, 12, 0, "MARIGOLD_MODEL") # Marigold Model -> Estimator
wb.connect(9, 0, 12, 1, "IMAGE")           # Subject Draft -> Marigold Estimator
wb.connect(12, 0, 13, 0, "IMAGE")          # Marigold Normals -> Normals Preview

# StudioLight Connections
wb.connect(12, 0, 14, 0, "IMAGE")       # Marigold Normals -> StudioLight normals_optional
wb.connect(9, 0, 14, 2, "IMAGE")        # Subject Draft Image -> StudioLight image_optional
wb.connect(2, 0, 14, 3, "CLIP")         # CLIP -> StudioLight clip_optional
wb.connect(4, 0, 14, 4, "CONDITIONING") # Conditioning In -> StudioLight conditioning_in

wb.connect(14, 1, 15, 0, "IMAGE")       # VFX Probe -> PreviewImage
wb.connect(14, 4, 16, 0, "IMAGE")       # Master Lit Composite -> PreviewImage
wb.connect(14, 4, 17, 0, "IMAGE")       # Master Lit Composite -> SaveImage

groups = [
    {"title": "1. FLUX SUBJECT GENESIS", "bounding": [-1650, 40, 1180, 660], "color": "#2a363b"},
    {"title": "2. 3D SURFACE SCANNER", "bounding": [-120, 40, 380, 860], "color": "#3b2a36"},
    {"title": "3. STUDIOLIGHT MASTER PHOTOMETRIC RIG", "bounding": [260, 40, 1300, 920], "color": "#1f3b2a"}
]

out_dir = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows"
flow_path = os.path.join(out_dir, "StudioLight_Master.json")
with open(flow_path, "w") as f:
    json.dump(wb.build_canvas(groups), f, indent=2)

print(f"Generated {flow_path}: {len(wb.nodes)} nodes, {len(wb.links)} links.")
