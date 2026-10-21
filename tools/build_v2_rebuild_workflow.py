#!/usr/bin/env python3
"""
ComfyUI-StudioLight — V2 Rebuild Workflow Generator
Generates: workflows/StudioLight_v2_Rebuild.json
Strictly implements the 5-Stage Rebuild Architecture:
1. Stage 1: Appearance Draft (Pass 1 - Soft readable scene, fully lit)
2. Stage 2: Pre-Flight Gate (Check landmark contrast & normal relief)
3. Stage 3: Masked Appearance Repair (Inpaint/lift face/rock BEFORE normals)
4. Stage 4: Physically Additive Proxy (L_proxy = L_ambient + (rho/pi)*E_flash + L_spec)
5. Stage 5: Pass 2 Denoise (0.55) (Polishes rich proxy into cinematic master)
"""

import json
import os

MASTER_PATH = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_Master.json"
OUTPUT_PATH = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_v2_Rebuild.json"

def build_v2_workflow():
    with open(MASTER_PATH, "r") as f:
        master = json.load(f)

    # Clone structure
    v2 = {
        "id": "StudioLight_v2_Rebuild",
        "revision": 1,
        "last_node_id": 100,
        "last_link_id": 200,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": master.get("config", {}),
        "extra": master.get("extra", {}),
        "version": 0.4
    }

    # Groups matching the 5 Stages
    v2["groups"] = [
        {"id": 1, "title": "STAGE 1: INPUTS & READABLE DRAFT", "bounding": [-2150, -350, 1350, 1150], "color": "#182836", "flags": {}},
        {"id": 2, "title": "STAGE 2 & 3: PRE-FLIGHT GATE & APPEARANCE REPAIR", "bounding": [-750, -350, 850, 1150], "color": "#332211", "flags": {}},
        {"id": 3, "title": "GEOMETRY EXTRACTION (NORMALS & DEPTH FROM REPAIRED DRAFT)", "bounding": [150, -350, 850, 1150], "color": "#223322", "flags": {}},
        {"id": 4, "title": "STAGE 4: PHYSICALLY ADDITIVE PROXY (ASTRA TRANSPORT)", "bounding": [1050, -350, 1050, 1150], "color": "#331828", "flags": {}},
        {"id": 5, "title": "STAGE 5: PASS 2 LIT SYNTHESIS (DENOISE 0.55)", "bounding": [2150, -350, 950, 1150], "color": "#183333", "flags": {}}
    ]

    # Node Definitions
    nodes = []
    links = []
    link_id = 1

    def add_link(src_node, src_slot, dst_node, dst_slot, link_type):
        nonlocal link_id
        current_id = link_id
        links.append([current_id, src_node, src_slot, dst_node, dst_slot, link_type])
        link_id += 1
        return current_id

    # -------------------------------------------------------------
    # 1. LOADERS & INPUTS
    # -------------------------------------------------------------
    nodes.append({
        "id": 1, "type": "UNETLoader", "pos": [-2100, -250], "size": [300, 80],
        "widgets_values": ["flux1-krea-dev.safetensors", "default"],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1, 2], "slot_index": 0}]
    })
    nodes.append({
        "id": 2, "type": "DualCLIPLoader", "pos": [-2100, -120], "size": [300, 100],
        "widgets_values": ["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux", "default"],
        "outputs": [{"name": "CLIP", "type": "CLIP", "links": [3, 4, 5], "slot_index": 0}]
    })
    nodes.append({
        "id": 3, "type": "VAELoader", "pos": [-2100, 20], "size": [300, 80],
        "widgets_values": ["ae.safetensors"],
        "outputs": [{"name": "VAE", "type": "VAE", "links": [6, 7, 8], "slot_index": 0}]
    })
    nodes.append({
        "id": 4, "type": "EmptyLatentImage", "pos": [-2100, 140], "size": [300, 100],
        "widgets_values": [1024, 1024, 1],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9], "slot_index": 0}]
    })
    nodes.append({
        "id": 6, "type": "PrimitiveStringMultiline", "pos": [-1750, -250], "size": [400, 180],
        "title": "Master Scene Prompt",
        "widgets_values": ["A muscular rugged caveman holding a burning wooden torch in a prehistoric cavern with rocky limestone walls, high detailed anatomy, realistic skin texture."],
        "outputs": [{"name": "STRING", "type": "STRING", "links": [10], "slot_index": 0}]
    })
    nodes.append({
        "id": 8, "type": "StudioPromptSanitizer", "pos": [-1750, -20], "size": [400, 120],
        "title": "StudioLight · Stage 1 Prompt Sanitizer",
        "widgets_values": ["auto"],
        "inputs": [{"name": "master_prompt", "type": "STRING", "link": 10}],
        "outputs": [
            {"name": "pass1_albedo_prompt", "type": "STRING", "links": [11], "slot_index": 0},
            {"name": "pass2_lit_prompt", "type": "STRING", "links": [12], "slot_index": 1}
        ]
    })
    nodes.append({
        "id": 9, "type": "CLIPTextEncode", "pos": [-1750, 150], "size": [400, 100],
        "title": "CLIP Encode (Pass 1 Readable Draft)",
        "inputs": [{"name": "clip", "type": "CLIP", "link": 3}, {"name": "text", "type": "STRING", "link": 11}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [13], "slot_index": 0}]
    })
    nodes.append({
        "id": 10, "type": "FluxGuidance", "pos": [-1750, 290], "size": [400, 80],
        "widgets_values": [3.5],
        "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 13}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [14], "slot_index": 0}]
    })
    nodes.append({
        "id": 5, "type": "CLIPTextEncode", "pos": [-1750, 410], "size": [400, 80],
        "title": "Negative Conditioning",
        "widgets_values": [""],
        "inputs": [{"name": "clip", "type": "CLIP", "link": 4}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [15, 16], "slot_index": 0}]
    })

    # -------------------------------------------------------------
    # STAGE 1: DRAFT SAMPLER (Steps 0 -> 10)
    # -------------------------------------------------------------
    nodes.append({
        "id": 11, "type": "KSamplerAdvanced", "pos": [-1300, -250], "size": [320, 300],
        "title": "Stage 1: Draft Sampler (Steps 0 -> 10)",
        "widgets_values": ["enable", 424242, "fixed", 30, 1.0, "euler", "simple", 0, 10, "disable"],
        "inputs": [
            {"name": "model", "type": "MODEL", "link": 1},
            {"name": "positive", "type": "CONDITIONING", "link": 14},
            {"name": "negative", "type": "CONDITIONING", "link": 15},
            {"name": "latent_image", "type": "LATENT", "link": 9}
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [17], "slot_index": 0}]
    })
    nodes.append({
        "id": 12, "type": "VAEDecode", "pos": [-1300, 90], "size": [220, 80],
        "title": "Decode Stage 1 Draft",
        "inputs": [{"name": "samples", "type": "LATENT", "link": 17}, {"name": "vae", "type": "VAE", "link": 6}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [18, 19, 20], "slot_index": 0}]
    })
    nodes.append({
        "id": 13, "type": "PreviewImage", "pos": [-1300, 210], "size": [320, 320],
        "title": "Preview: Stage 1 Readable Draft",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 18}]
    })

    # -------------------------------------------------------------
    # STAGE 2: PRE-FLIGHT GATE
    # -------------------------------------------------------------
    nodes.append({
        "id": 50, "type": "StudioTransportPreflight", "pos": [-700, -250], "size": [380, 220],
        "title": "Stage 2: Pre-Flight Gate Review",
        "widgets_values": ["encoded_x_right_y_up_z_toward_camera", 0, 0, 0, 0],
        "inputs": [
            {"name": "draft", "type": "IMAGE", "link": 19},
            {"name": "normals", "type": "IMAGE", "link": 21} # Connects from DSINE
        ],
        "outputs": [
            {"name": "review_overlay", "type": "IMAGE", "links": [22], "slot_index": 0},
            {"name": "review_report", "type": "STRING", "links": [23], "slot_index": 1}
        ]
    })
    nodes.append({
        "id": 51, "type": "PreviewImage", "pos": [-700, 10], "size": [380, 380],
        "title": "Preview: Stage 2 Preflight Diagnostic",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 22}]
    })

    # -------------------------------------------------------------
    # STAGE 3: MASKED APPEARANCE REPAIR
    # -------------------------------------------------------------
    nodes.append({
        "id": 52, "type": "StudioAppearanceRepair", "pos": [-700, 420], "size": [380, 200],
        "title": "Stage 3: Masked Appearance Repair",
        "widgets_values": ["auto_adaptive_lift", 16, 0.45],
        "inputs": [
            {"name": "draft", "type": "IMAGE", "link": 20},
            {"name": "review_report", "type": "STRING", "link": 23}
        ],
        "outputs": [
            {"name": "repaired_draft", "type": "IMAGE", "links": [24, 25, 26, 27], "slot_index": 0},
            {"name": "repair_mask", "type": "MASK", "links": [28], "slot_index": 1},
            {"name": "repair_status", "type": "STRING", "links": [], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 53, "type": "PreviewImage", "pos": [-700, 650], "size": [380, 320],
        "title": "Preview: Stage 3 Accepted Draft",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 24}]
    })

    # -------------------------------------------------------------
    # GEOMETRY EXTRACTION (NORMALS & DEPTH FROM REPAIRED DRAFT)
    # -------------------------------------------------------------
    nodes.append({
        "id": 14, "type": "DSINE-NormalMapPreprocessor", "pos": [200, -250], "size": [350, 120],
        "title": "Extract 3D Normals (DSINE)",
        "widgets_values": [60, 5, 1024],
        "inputs": [{"name": "image", "type": "IMAGE", "link": 25}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [21, 29, 30], "slot_index": 0}]
    })
    nodes.append({
        "id": 31, "type": "PreviewImage", "pos": [200, -90], "size": [350, 320],
        "title": "Preview: DSINE 3D Normals",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 29}]
    })
    nodes.append({
        "id": 29, "type": "DepthAnythingV2Preprocessor", "pos": [200, 270], "size": [350, 120],
        "title": "Extract Depth (Depth Anything V2)",
        "widgets_values": ["depth_anything_v2_vitl.pth", 1024],
        "inputs": [{"name": "image", "type": "IMAGE", "link": 26}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [31, 32], "slot_index": 0}]
    })
    nodes.append({
        "id": 30, "type": "PreviewImage", "pos": [200, 430], "size": [350, 320],
        "title": "Preview: Depth Map",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 31}]
    })

    # -------------------------------------------------------------
    # STAGE 4: PHYSICALLY ADDITIVE PROXY (ASTRA TRANSPORT)
    # -------------------------------------------------------------
    nodes.append({
        "id": 54, "type": "StudioTransportGeometry", "pos": [1100, -250], "size": [380, 200],
        "title": "StudioLight · Assumed Geometry",
        "widgets_values": ["Inside", "In-Between", "Meters (Depth)", 6.0, 2.0, 50.0, 0.0, 0.0],
        "inputs": [
            {"name": "depth", "type": "IMAGE", "link": 32},
            {"name": "subject_mask", "type": "MASK", "link": 28}
        ],
        "outputs": [
            {"name": "geometry", "type": "SL_GEOMETRY", "links": [33], "slot_index": 0},
            {"name": "assumed_depth_preview", "type": "IMAGE", "links": [34], "slot_index": 1},
            {"name": "report", "type": "STRING", "links": [], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 55, "type": "StudioTransportWall", "pos": [1100, -10], "size": [380, 260],
        "title": "StudioLight · Cavity Wall Transport",
        "widgets_values": ["encoded_x_right_y_up_z_toward_camera", 0.0, 0.0, 1.0, 40.0, 30.0, 20.0, 0.2],
        "inputs": [
            {"name": "geometry", "type": "SL_GEOMETRY", "link": 33},
            {"name": "normals", "type": "IMAGE", "link": 30},
            {"name": "receiver_mask", "type": "MASK", "link": 28},
            {"name": "visibility", "type": "MASK", "link": 28}
        ],
        "outputs": [
            {"name": "flash_transport", "type": "SL_FLASH", "links": [35], "slot_index": 0},
            {"name": "irradiance_heatmap", "type": "IMAGE", "links": [36], "slot_index": 1},
            {"name": "valid_geometry", "type": "MASK", "links": [], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 56, "type": "StudioTransportAmbient", "pos": [1100, 280], "size": [380, 160],
        "title": "StudioLight · Ambient Baseline",
        "widgets_values": [-3.0, 1.0],
        "inputs": [
            {"name": "readable_draft", "type": "IMAGE", "link": 27},
            {"name": "ambient_tint", "type": "IMAGE", "link": 27}
        ],
        "outputs": [
            {"name": "ambient_linear", "type": "SL_LINEAR", "links": [37], "slot_index": 0},
            {"name": "draft_linear", "type": "SL_LINEAR", "links": [38], "slot_index": 1},
            {"name": "ambient_preview", "type": "IMAGE", "links": [39], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 57, "type": "StudioTransportReflectance", "pos": [1100, 470], "size": [380, 140],
        "title": "StudioLight · Reflectance Prior",
        "widgets_values": [False],
        "inputs": [
            {"name": "draft_linear", "type": "SL_LINEAR", "link": 38},
            {"name": "prior_plate", "type": "IMAGE", "link": 27},
            {"name": "mask", "type": "MASK", "link": 28}
        ],
        "outputs": [
            {"name": "reflectance_linear", "type": "SL_LINEAR", "links": [40], "slot_index": 0},
            {"name": "reflectance_preview", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "unresolved_mask", "type": "MASK", "links": [], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 58, "type": "StudioTransportComposite", "pos": [1550, -250], "size": [380, 180],
        "title": "Stage 4: Additive Flash Composite",
        "widgets_values": [1.0, 1.0, "reinhard"],
        "inputs": [
            {"name": "ambient_linear", "type": "SL_LINEAR", "link": 37},
            {"name": "reflectance_linear", "type": "SL_LINEAR", "link": 40},
            {"name": "flash_transport", "type": "SL_FLASH", "link": 35}
        ],
        "outputs": [
            {"name": "proxy_linear", "type": "SL_LINEAR", "links": [], "slot_index": 0},
            {"name": "proxy_image", "type": "IMAGE", "links": [41, 42], "slot_index": 1},
            {"name": "flash_heatmap", "type": "IMAGE", "links": [43], "slot_index": 2}
        ]
    })
    nodes.append({
        "id": 19, "type": "PreviewImage", "pos": [1550, -30], "size": [450, 450],
        "title": "Preview: Stage 4 Additive Proxy Plate",
        "inputs": [{"name": "images", "type": "IMAGE", "link": 41}]
    })

    # -------------------------------------------------------------
    # STAGE 5: PASS 2 LIT SYNTHESIS (DENOISE 0.55)
    # -------------------------------------------------------------
    nodes.append({
        "id": 20, "type": "VAEEncode", "pos": [2200, -250], "size": [220, 80],
        "title": "VAE Encode Stage 4 Proxy",
        "inputs": [{"name": "pixels", "type": "IMAGE", "link": 42}, {"name": "vae", "type": "VAE", "link": 7}],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [44], "slot_index": 0}]
    })
    nodes.append({
        "id": 23, "type": "CLIPTextEncode", "pos": [2200, -130], "size": [400, 120],
        "title": "CLIP Encode (Pass 2 Lit Prompt)",
        "inputs": [{"name": "clip", "type": "CLIP", "link": 5}, {"name": "text", "type": "STRING", "link": 12}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [45], "slot_index": 0}]
    })
    nodes.append({
        "id": 24, "type": "FluxGuidance", "pos": [2200, 30], "size": [400, 80],
        "widgets_values": [3.5],
        "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 45}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [46], "slot_index": 0}]
    })
    nodes.append({
        "id": 25, "type": "KSampler", "pos": [2200, 150], "size": [320, 260],
        "title": "Stage 5: Pass 2 KSampler (CALIBRATION LEVER)",
        "widgets_values": [424242, "fixed", 30, 1.0, "euler", "simple", 0.55],
        "inputs": [
            {"name": "model", "type": "MODEL", "link": 2},
            {"name": "positive", "type": "CONDITIONING", "link": 46},
            {"name": "negative", "type": "CONDITIONING", "link": 16},
            {"name": "latent_image", "type": "LATENT", "link": 44}
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": [47], "slot_index": 0}]
    })
    nodes.append({
        "id": 26, "type": "VAEDecode", "pos": [2650, 150], "size": [220, 80],
        "title": "Decode Final Master",
        "inputs": [{"name": "samples", "type": "LATENT", "link": 47}, {"name": "vae", "type": "VAE", "link": 8}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [48, 49], "slot_index": 0}]
    })
    nodes.append({
        "id": 27, "type": "SaveImage", "pos": [2650, 270], "size": [400, 420],
        "title": "💎 FINAL LIT RENDER",
        "widgets_values": ["StudioLight_v2_Master"],
        "inputs": [{"name": "images", "type": "IMAGE", "link": 48}]
    })

    # Assemble all links
    link_definitions = [
        [1, 1, 0, 11, 0, "MODEL"],
        [2, 1, 0, 25, 0, "MODEL"],
        [3, 2, 0, 9, 0, "CLIP"],
        [4, 2, 0, 5, 0, "CLIP"],
        [5, 2, 0, 23, 0, "CLIP"],
        [6, 3, 0, 12, 1, "VAE"],
        [7, 3, 0, 20, 1, "VAE"],
        [8, 3, 0, 26, 1, "VAE"],
        [9, 4, 0, 11, 3, "LATENT"],
        [10, 6, 0, 8, 0, "STRING"],
        [11, 8, 0, 9, 1, "STRING"],
        [12, 8, 1, 23, 1, "STRING"],
        [13, 9, 0, 10, 0, "CONDITIONING"],
        [14, 10, 0, 11, 1, "CONDITIONING"],
        [15, 5, 0, 11, 2, "CONDITIONING"],
        [16, 5, 0, 25, 2, "CONDITIONING"],
        [17, 11, 0, 12, 0, "LATENT"],
        [18, 12, 0, 13, 0, "IMAGE"],
        [19, 12, 0, 50, 0, "IMAGE"],
        [20, 12, 0, 52, 0, "IMAGE"],
        [21, 14, 0, 50, 1, "IMAGE"],
        [22, 50, 0, 51, 0, "IMAGE"],
        [23, 50, 1, 52, 1, "STRING"],
        [24, 52, 0, 53, 0, "IMAGE"],
        [25, 52, 0, 14, 0, "IMAGE"],
        [26, 52, 0, 29, 0, "IMAGE"],
        [27, 52, 0, 56, 0, "IMAGE"],
        [28, 52, 1, 54, 1, "MASK"],
        [29, 14, 0, 31, 0, "IMAGE"],
        [30, 14, 0, 55, 1, "IMAGE"],
        [31, 29, 0, 30, 0, "IMAGE"],
        [32, 29, 0, 54, 0, "IMAGE"],
        [33, 54, 0, 55, 0, "SL_GEOMETRY"],
        [34, 54, 1, 19, 0, "IMAGE"],
        [35, 55, 0, 58, 2, "SL_FLASH"],
        [36, 55, 1, 19, 0, "IMAGE"],
        [37, 56, 0, 58, 0, "SL_LINEAR"],
        [38, 56, 1, 57, 0, "SL_LINEAR"],
        [39, 56, 2, 19, 0, "IMAGE"],
        [40, 57, 0, 58, 1, "SL_LINEAR"],
        [41, 58, 1, 19, 0, "IMAGE"],
        [42, 58, 1, 20, 0, "IMAGE"],
        [43, 58, 2, 19, 0, "IMAGE"],
        [44, 20, 0, 25, 3, "LATENT"],
        [45, 23, 0, 24, 0, "CONDITIONING"],
        [46, 24, 0, 25, 1, "CONDITIONING"],
        [47, 25, 0, 26, 0, "LATENT"],
        [48, 26, 0, 27, 0, "IMAGE"],
        [49, 26, 0, 19, 0, "IMAGE"]
    ]

    v2["nodes"] = nodes
    v2["links"] = link_definitions
    v2["last_node_id"] = max(n["id"] for n in nodes)
    v2["last_link_id"] = len(link_definitions)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(v2, f, indent=2)

    print(f"✓ Successfully built separate V2 Rebuild workflow at: {OUTPUT_PATH}")
    print(f" - Total nodes: {len(nodes)}")
    print(f" - Total links: {len(link_definitions)}")
    print(f" - Stages: 5 fully wired stages matching the approved architecture diagram.")

if __name__ == "__main__":
    build_v2_workflow()
