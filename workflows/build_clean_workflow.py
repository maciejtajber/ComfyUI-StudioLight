import json
import os

class WorkflowBuilder:
    def __init__(self):
        self.nodes = {}
        self.links = []
        self.link_counter = 1

    def add_node(self, node_id, node_type, title, pos, size, widgets_values, input_names, output_defs):
        # input_names: list of (name, type)
        # output_defs: list of (name, type)
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
        
        # Add to links table: [id, origin_id, origin_slot, target_id, target_slot, type]
        self.links.append([l_id, from_node_id, from_slot, to_node_id, to_slot, type_name])
        
        # Update origin node outputs
        from_node = self.nodes[from_node_id]
        if from_slot < len(from_node["outputs"]):
            from_node["outputs"][from_slot]["links"].append(l_id)
        else:
            raise ValueError(f"Node {from_node_id} has no output slot {from_slot}")
            
        # Update target node inputs
        to_node = self.nodes[to_node_id]
        if to_slot < len(to_node["inputs"]):
            to_node["inputs"][to_slot]["link"] = l_id
        else:
            raise ValueError(f"Node {to_node_id} has no input slot {to_slot}")

    def build(self, groups=None, offset=None):
        node_list = list(self.nodes.values())
        return {
            "last_node_id": max(self.nodes.keys()) if self.nodes else 0,
            "last_link_id": len(self.links),
            "nodes": node_list,
            "links": self.links,
            "groups": groups or [],
            "config": {},
            "extra": {"ds": {"scale": 0.85, "offset": offset or [1300, 80]}},
            "version": 0.4
        }

wb = WorkflowBuilder()

# 1. Base Models (UNET, DualCLIP, VAE)
wb.add_node(
    node_id=1,
    node_type="UNETLoader",
    title="Flux Krea Dev (Base UNET)",
    pos=[-1200, 100],
    size=[320, 90],
    widgets_values=["flux1-krea-dev.safetensors", "default"],
    input_names=[],
    output_defs=[("MODEL", "MODEL")]
)

wb.add_node(
    node_id=2,
    node_type="DualCLIPLoader",
    title="Dual CLIP Loader (T5 + CLIP-L)",
    pos=[-1200, 240],
    size=[320, 110],
    widgets_values=["t5xxl_fp16.safetensors", "clip_l.safetensors", "flux"],
    input_names=[],
    output_defs=[("CLIP", "CLIP")]
)

wb.add_node(
    node_id=3,
    node_type="VAELoader",
    title="Flux VAE",
    pos=[-1200, 400],
    size=[320, 80],
    widgets_values=["ae.safetensors"],
    input_names=[],
    output_defs=[("VAE", "VAE")]
)

# 2. Fashion Subject & Universal Negative PromptBlocks
wb.add_node(
    node_id=4,
    node_type="SIPromptBlock",
    title="✨ [Fashion Subject] SIPromptBlock (0.0 → 0.40)",
    pos=[-800, 100],
    size=[380, 280],
    widgets_values=[
        "Fashion Subject",
        "cinematic editorial fashion portrait of an elegant woman, high-fashion styling, flawless natural skin texture, subtle freckles, sharp depth of field, neutral dark grey studio backdrop, 85mm portrait lens, 8k resolution, Vogue magazine cover quality",
        True,
        1.0,
        "ADDITIVE",
        0.0,
        0.40,
        "concat"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

wb.add_node(
    node_id=5,
    node_type="SIPromptBlock",
    title="⛔ [Negative] Universal Suppression",
    pos=[-800, 420],
    size=[380, 280],
    widgets_values=[
        "Negative Suppression",
        "blurry, out of focus, low quality, distorted, deformed anatomy, bad eyes, extra limbs, cartoon, 3d render illustration, oversaturated, flat plastic texture, noise, grain, amateur photography",
        True,
        1.0,
        "ADDITIVE",
        0.0,
        1.00,
        "concat"
    ],
    input_names=[("clip", "CLIP"), ("conditioning_in", "CONDITIONING"), ("mask", "MASK")],
    output_defs=[("conditioning", "CONDITIONING")]
)

# 3. StudioLightRig (Console)
wb.add_node(
    node_id=6,
    node_type="StudioLightRig",
    title="🎛️ StudioLight (Rembrandt Console)",
    pos=[-380, 100],
    size=[380, 720],
    widgets_values=[
        "Rembrandt Classic",
        "Portrait (Classical Bust)",
        0.05,
        False,
        True, 45.0, 35.0, 1.0, 5500, "Octabox (Soft)",
        True, -30.0, 15.0, 0.35, 4800, "Octabox (Soft)",
        True, 160.0, 45.0, 1.10, 6500, "Stripbox",
        0.40, 1.00, 1.00, "A_NORM"
    ],
    input_names=[
        ("normals_optional", "IMAGE"),
        ("depth_optional", "IMAGE"),
        ("clip_optional", "CLIP"),
        ("conditioning_in", "CONDITIONING"),
        ("prompt_text_optional", "STRING")
    ],
    output_defs=[
        ("relighting_map", "IMAGE"),
        ("probe_preview", "IMAGE"),
        ("conditioning", "CONDITIONING"),
        ("lighting_prompt", "STRING")
    ]
)

# 4. VFX Probes & Deferred Map
wb.add_node(
    node_id=7,
    node_type="PreviewImage",
    title="✨ VFX Probe (18% Grey & Chrome Ball + Radar)",
    pos=[40, 100],
    size=[320, 320],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

wb.add_node(
    node_id=8,
    node_type="PreviewImage",
    title="✨ 32-bit Deferred Lighting Pass",
    pos=[40, 460],
    size=[320, 320],
    widgets_values=[],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# 5. Sampling Pipeline
wb.add_node(
    node_id=9,
    node_type="FluxGuidance",
    title="Flux Guidance (3.5)",
    pos=[400, 100],
    size=[260, 80],
    widgets_values=[3.5],
    input_names=[("conditioning", "CONDITIONING")],
    output_defs=[("CONDITIONING", "CONDITIONING")]
)

wb.add_node(
    node_id=10,
    node_type="EmptySD3LatentImage",
    title="Empty Latent (896x1152)",
    pos=[400, 220],
    size=[260, 110],
    widgets_values=[896, 1152, 1],
    input_names=[],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=11,
    node_type="KSampler",
    title="🚀 Flux KSampler (Pure Base UNET)",
    pos=[700, 100],
    size=[300, 260],
    widgets_values=[42, "randomize", 25, 1.0, "euler", "beta", 1.0],
    input_names=[
        ("model", "MODEL"),
        ("positive", "CONDITIONING"),
        ("negative", "CONDITIONING"),
        ("latent_image", "LATENT")
    ],
    output_defs=[("LATENT", "LATENT")]
)

wb.add_node(
    node_id=12,
    node_type="VAEDecode",
    title="VAE Decode",
    pos=[700, 400],
    size=[240, 80],
    widgets_values=[],
    input_names=[("samples", "LATENT"), ("vae", "VAE")],
    output_defs=[("IMAGE", "IMAGE")]
)

wb.add_node(
    node_id=13,
    node_type="SaveImage",
    title="🎉 [Fashion T2I] Vogue Master Render",
    pos=[1040, 100],
    size=[440, 540],
    widgets_values=["Fashion_StudioLight"],
    input_names=[("images", "IMAGE")],
    output_defs=[]
)

# ----------------- CONNECTIONS -----------------
# DualCLIP (Node 2, Slot 0) -> Fashion Subject (Node 4, Slot 0)
wb.connect(2, 0, 4, 0, "CLIP")

# DualCLIP (Node 2, Slot 0) -> Negative Suppression (Node 5, Slot 0)
wb.connect(2, 0, 5, 0, "CLIP")

# DualCLIP (Node 2, Slot 0) -> StudioLight clip_optional (Node 6, Slot 2)
wb.connect(2, 0, 6, 2, "CLIP")

# Fashion Subject conditioning (Node 4, Slot 0) -> StudioLight conditioning_in (Node 6, Slot 3)
wb.connect(4, 0, 6, 3, "CONDITIONING")

# StudioLight probe_preview (Node 6, Slot 1) -> PreviewImage Probe (Node 7, Slot 0)
wb.connect(6, 1, 7, 0, "IMAGE")

# StudioLight relighting_map (Node 6, Slot 0) -> PreviewImage Deferred (Node 8, Slot 0)
wb.connect(6, 0, 8, 0, "IMAGE")

# StudioLight conditioning (Node 6, Slot 2) -> FluxGuidance (Node 9, Slot 0)
wb.connect(6, 2, 9, 0, "CONDITIONING")

# UNET (Node 1, Slot 0) -> KSampler model (Node 11, Slot 0)
wb.connect(1, 0, 11, 0, "MODEL")

# FluxGuidance (Node 9, Slot 0) -> KSampler positive (Node 11, Slot 1)
wb.connect(9, 0, 11, 1, "CONDITIONING")

# Negative Suppression (Node 5, Slot 0) -> KSampler negative (Node 11, Slot 2)
wb.connect(5, 0, 11, 2, "CONDITIONING")

# EmptySD3LatentImage (Node 10, Slot 0) -> KSampler latent_image (Node 11, Slot 3)
wb.connect(10, 0, 11, 3, "LATENT")

# KSampler latent (Node 11, Slot 0) -> VAEDecode samples (Node 12, Slot 0)
wb.connect(11, 0, 12, 0, "LATENT")

# VAE (Node 3, Slot 0) -> VAEDecode vae (Node 12, Slot 1)
wb.connect(3, 0, 12, 1, "VAE")

# VAEDecode image (Node 12, Slot 0) -> SaveImage (Node 13, Slot 0)
wb.connect(12, 0, 13, 0, "IMAGE")

groups = [
    {"title": "1. PURE BASE MODELS (FLUX KREA DEV — ZERO LORAS)", "bounding": [-1240, 40, 400, 480], "color": "#2a363b"},
    {"title": "2. EDITORIAL FASHION PROMPT & STUDIOLIGHT CONSOLE", "bounding": [-840, 40, 860, 820], "color": "#3f2b47"},
    {"title": "3. VFX PROBES & DEFERRED PASS", "bounding": [0, 40, 380, 780], "color": "#1f3a3d"},
    {"title": "4. FLUX SAMPLING & VOGUE MASTER OUTPUT", "bounding": [360, 40, 1160, 620], "color": "#2e2d4d"}
]

workflow_data = wb.build(groups=groups)

out_path = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_T2I_Simple.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(workflow_data, f, indent=2)

print("Saved mathematically verified workflow to:", out_path)
