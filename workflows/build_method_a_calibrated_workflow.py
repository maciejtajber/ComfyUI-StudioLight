"""
Builds StudioLight_MethodA_Calibrated.json
Production-grade Method A (Stochastic Re-Diffusion) calibration workflow.
Features:
- Dual-Conditioning Prompt Engine (Pass 1 shadowless, Pass 2 dynamic lighting sync)
- DSINE 3D Normals + StudioLightRig (3-point console)
- StudioPhotometricComposite (specular_kick: 0.0 for black leather preservation)
- SIShadowHighlight (filmic tone dynamic range)
- VAEEncode into Pass 2 KSampler (30 steps, Euler, customizable scheduler & denoise)
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
            "title": title,
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

    def build(self, groups=None):
        return {
            "last_node_id": max(self.nodes.keys()) if self.nodes else 0,
            "last_link_id": len(self.links),
            "nodes": list(self.nodes.values()),
            "links": self.links,
            "groups": groups or [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }

def build_method_a():
    wb = WorkflowBuilder()

    # --- ZONE 1: CORE FLUX LOADERS ---
    wb.add_node(
        node_id=1,
        node_type="UNETLoader",
        title="Flux Krea Dev (Base UNET)",
        pos=[-1400, -260],
        size=[320, 82],
        widgets_values=["flux1-krea-dev.safetensors", "default"],
        input_names=[],
        output_defs=[("MODEL", "MODEL")],
    )

    wb.add_node(
        node_id=2,
        node_type="DualCLIPLoader",
        title="Dual CLIP Loader (T5xxl + CLIP-L)",
        pos=[-1400, -140],
        size=[320, 106],
        widgets_values=["clip_l.safetensors", "t5xxl_fp16.safetensors", "flux", "default"],
        input_names=[],
        output_defs=[("CLIP", "CLIP")],
    )

    wb.add_node(
        node_id=3,
        node_type="VAELoader",
        title="Flux VAE",
        pos=[-1400, 0],
        size=[320, 58],
        widgets_values=["ae.safetensors"],
        input_names=[],
        output_defs=[("VAE", "VAE")],
    )

    wb.add_node(
        node_id=4,
        node_type="EmptyLatentImage",
        title="Empty Latent (1024x1024)",
        pos=[-1400, 100],
        size=[320, 106],
        widgets_values=[1024, 1024, 1],
        input_names=[],
        output_defs=[("LATENT", "LATENT")],
    )

    # Negative conditioning (Flux standard empty)
    wb.add_node(
        node_id=5,
        node_type="CLIPTextEncode",
        title="Negative Conditioning (Empty)",
        pos=[-1400, 240],
        size=[320, 80],
        widgets_values=[""],
        input_names=[("clip", "CLIP")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    # --- PROMPT ENGINE (DUAL-CONDITIONING) ---
    subject_prompt = (
        "High fashion editorial portrait of a young elegant woman holding a glossy black patent leather handbag, "
        "wearing a sharp tailored wool coat, architectural background, sharp focus, 8k, raw photographic textures"
    )
    wb.add_node(
        node_id=6,
        node_type="PrimitiveStringMultiline",
        title="Base Subject Prompt (Clean Content)",
        pos=[-1040, -260],
        size=[320, 140],
        widgets_values=[subject_prompt],
        input_names=[],
        output_defs=[("STRING", "STRING")],
    )

    wb.add_node(
        node_id=7,
        node_type="PrimitiveStringMultiline",
        title="Pass 1 Albedo Lighting Suffix (Shadowless)",
        pos=[-1040, -90],
        size=[320, 90],
        widgets_values=[", flat diffuse high-key ambient lighting, shadowless, Vogue cover style"],
        input_names=[],
        output_defs=[("STRING", "STRING")],
    )

    wb.add_node(
        node_id=8,
        node_type="StringConcatenate",
        title="Concat Pass 1 (Subject + Shadowless)",
        pos=[-1040, 30],
        size=[320, 80],
        widgets_values=[],
        input_names=[("string_a", "STRING"), ("string_b", "STRING")],
        output_defs=[("STRING", "STRING")],
    )

    wb.add_node(
        node_id=9,
        node_type="CLIPTextEncode",
        title="CLIP Text Encode (Pass 1 Albedo)",
        pos=[-1040, 140],
        size=[320, 100],
        widgets_values=[""],
        input_names=[("clip", "CLIP"), ("text", "STRING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    wb.add_node(
        node_id=10,
        node_type="FluxGuidance",
        title="Flux Guidance Pass 1 (3.5)",
        pos=[-1040, 270],
        size=[320, 58],
        widgets_values=[3.5],
        input_names=[("conditioning", "CONDITIONING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    # --- ZONE 2: PASS 1 DRAFT GENERATION (STEPS 0 -> 10) ---
    wb.add_node(
        node_id=11,
        node_type="KSamplerAdvanced",
        title="Pass 1 Draft Sampler (Steps 0 -> 10)",
        pos=[-680, -260],
        size=[320, 280],
        widgets_values=[
            "enable",   # add_noise
            424242,     # noise_seed
            "fixed",    # control_after_generate
            30,         # steps
            1.0,        # cfg
            "euler",    # sampler_name
            "simple",   # scheduler
            0,          # start_at_step
            10,         # end_at_step
            "disable",  # return_with_leftover_noise
        ],
        input_names=[
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ],
        output_defs=[("LATENT", "LATENT")],
    )

    wb.add_node(
        node_id=12,
        node_type="VAEDecode",
        title="Decode Pass 1 Draft",
        pos=[-680, 60],
        size=[200, 60],
        widgets_values=[],
        input_names=[("samples", "LATENT"), ("vae", "VAE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=13,
        node_type="PreviewImage",
        title="Preview: Pass 1 Neutral Draft (Step 10)",
        pos=[-680, 160],
        size=[320, 320],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # --- ZONE 3: 3D NORMAL EXTRACTION & PHOTOMETRIC COMPOSITING ---
    wb.add_node(
        node_id=14,
        node_type="DSINE-NormalMapPreprocessor",
        title="Extract 3D Normals (DSINE)",
        pos=[-320, -260],
        size=[280, 140],
        widgets_values=[60.0, 5, 1024],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=15,
        node_type="StudioLightRig",
        title="StudioLightRig (3-Point Photographic Console)",
        pos=[-320, -90],
        size=[340, 520],
        widgets_values=[
            "Rembrandt Classic",
            "American Plan (3/4 Shot)",
            0.04,
            False,
            True, 45.0, 30.0, 1.25, 5500, "Beauty Dish (Silver)",   # Key
            True, -45.0, 15.0, 0.35, 4800, "Octabox (Soft)",        # Fill
            True, 140.0, 40.0, 1.20, 6800, "Stripbox",               # Rim
        ],
        input_names=[
            ("scene_3d", "SCENE_3D"),
            ("normals_optional", "IMAGE"),
            ("depth_optional", "IMAGE"),
        ],
        output_defs=[
            ("relighting_map", "IMAGE"),
            ("probe_preview", "IMAGE"),
            ("lighting_prompt", "STRING"),
        ],
    )

    wb.add_node(
        node_id=16,
        node_type="PreviewImage",
        title="VFX Probes & Studio Radar",
        pos=[-320, 460],
        size=[340, 220],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    wb.add_node(
        node_id=17,
        node_type="StudioPhotometricComposite",
        title="Studio Photometric Composite (Linear Rec.709)",
        pos=[60, -260],
        size=[300, 220],
        widgets_values=[
            1.0,    # lighting_strength
            0.05,   # ambient_floor
            0.0,    # specular_kick (LOCKED TO 0.0 - PREVENTS MILKY LEATHER BLOWOUT!)
            0.40,   # chroma_bleed
            3.5,    # filmic_white_point
        ],
        input_names=[
            ("image", "IMAGE"),
            ("lighting_map", "IMAGE"),
        ],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=18,
        node_type="SIShadowHighlight",
        title="SI Shadow & Highlight Sculptor",
        pos=[60, 0],
        size=[300, 180],
        widgets_values=[0.20, 0.10, 0.50, 0.50, 1.05],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=19,
        node_type="PreviewImage",
        title="Preview: 3D Lit Proxy Plate (Prior to FLUX)",
        pos=[60, 220],
        size=[300, 300],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    wb.add_node(
        node_id=20,
        node_type="VAEEncode",
        title="VAE Encode Lit Plate",
        pos=[60, 550],
        size=[200, 60],
        widgets_values=[],
        input_names=[("pixels", "IMAGE"), ("vae", "VAE")],
        output_defs=[("LATENT", "LATENT")],
    )

    # --- ZONE 4: PASS 2 DUAL-CONDITIONING PROMPT WIRING ---
    wb.add_node(
        node_id=21,
        node_type="StringConcatenate",
        title="Concat Pass 2 (Subject + Dynamic Lighting)",
        pos=[400, -260],
        size=[300, 80],
        widgets_values=[],
        input_names=[("string_a", "STRING"), ("string_b", "STRING")],
        output_defs=[("STRING", "STRING")],
    )

    wb.add_node(
        node_id=22,
        node_type="PreviewAny",
        title="Preview Pass 2 Synchronized Prompt",
        pos=[400, -150],
        size=[300, 120],
        widgets_values=[],
        input_names=[("source", "STRING")],
        output_defs=[("output", "STRING")],
    )

    wb.add_node(
        node_id=23,
        node_type="CLIPTextEncode",
        title="CLIP Text Encode (Pass 2 Lit Prompt)",
        pos=[400, 0],
        size=[300, 100],
        widgets_values=[""],
        input_names=[("clip", "CLIP"), ("text", "STRING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    wb.add_node(
        node_id=24,
        node_type="FluxGuidance",
        title="Flux Guidance Pass 2 (3.5)",
        pos=[400, 130],
        size=[300, 58],
        widgets_values=[3.5],
        input_names=[("conditioning", "CONDITIONING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    # --- ZONE 5: PASS 2 STOCHASTIC RE-DIFFUSION (METHOD A ENGINE) ---
    wb.add_node(
        node_id=25,
        node_type="KSampler",
        title="Pass 2 KSampler (CALIBRATION CONTROL LEVER)",
        pos=[740, -260],
        size=[340, 300],
        widgets_values=[
            424242,     # seed (fixed for apples-to-apples baseline)
            "fixed",    # control_after_generate
            30,         # steps
            1.0,        # cfg
            "euler",    # sampler_name
            "simple",   # scheduler (baseline: simple, test against beta!)
            0.72,       # denoise (the calibration bracket: 0.70 / 0.72 / 0.74)
        ],
        input_names=[
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ],
        output_defs=[("LATENT", "LATENT")],
    )

    wb.add_node(
        node_id=26,
        node_type="VAEDecode",
        title="Decode Final Master",
        pos=[1120, -260],
        size=[200, 60],
        widgets_values=[],
        input_names=[("samples", "LATENT"), ("vae", "VAE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=27,
        node_type="SaveImage",
        title="💎 FINAL LIT RENDER (Method A Calibrated)",
        pos=[1120, -160],
        size=[360, 100],
        widgets_values=["MethodA_Calibrated_Final"],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    wb.add_node(
        node_id=28,
        node_type="PreviewImage",
        title="Preview: Final Lit Result",
        pos=[1120, -30],
        size=[360, 400],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # --- WIRING LINKS ---
    # Zone 1 Loaders
    wb.connect(2, 0, 5, 0, "CLIP")      # DualCLIP -> Negative
    wb.connect(2, 0, 9, 0, "CLIP")      # DualCLIP -> Pass 1 CLIPText
    wb.connect(2, 0, 23, 0, "CLIP")     # DualCLIP -> Pass 2 CLIPText

    # Prompt Concat Pass 1
    wb.connect(6, 0, 8, 0, "STRING")    # Clean Subject -> Concat Pass 1 (A)
    wb.connect(7, 0, 8, 1, "STRING")    # Shadowless -> Concat Pass 1 (B)
    wb.connect(8, 0, 9, 1, "STRING")    # Concat Pass 1 -> CLIPTextEncode
    wb.connect(9, 0, 10, 0, "CONDITIONING") # Pass 1 CLIPText -> FluxGuidance

    # Pass 1 Draft Sampler
    wb.connect(1, 0, 11, 0, "MODEL")    # UNET -> Pass 1 Sampler
    wb.connect(10, 0, 11, 1, "CONDITIONING") # Pass 1 Pos -> Pass 1 Sampler
    wb.connect(5, 0, 11, 2, "CONDITIONING")  # Neg -> Pass 1 Sampler
    wb.connect(4, 0, 11, 3, "LATENT")   # Empty Latent -> Pass 1 Sampler

    # Pass 1 Decode & Preview
    wb.connect(11, 0, 12, 0, "LATENT")  # Pass 1 Latent -> VAEDecode
    wb.connect(3, 0, 12, 1, "VAE")      # VAE -> VAEDecode
    wb.connect(12, 0, 13, 0, "IMAGE")   # Draft Image -> Preview

    # Zone 3: DSINE & StudioLightRig
    wb.connect(12, 0, 14, 0, "IMAGE")   # Draft Image -> DSINE
    wb.connect(14, 0, 15, 1, "IMAGE")   # DSINE Normals -> StudioLightRig normals_optional
    wb.connect(15, 1, 16, 0, "IMAGE")   # Probe Preview -> PreviewImage

    # Zone 3: Photometric Composite & Sculptor
    wb.connect(12, 0, 17, 0, "IMAGE")   # Draft Image -> Composite
    wb.connect(15, 0, 17, 1, "IMAGE")   # Relighting Map -> Composite
    wb.connect(17, 0, 18, 0, "IMAGE")   # Composite -> SIShadowHighlight
    wb.connect(18, 0, 19, 0, "IMAGE")   # Sculpted Lit Image -> Preview
    wb.connect(18, 0, 20, 0, "IMAGE")   # Sculpted Lit Image -> VAEEncode
    wb.connect(3, 0, 20, 1, "VAE")      # VAE -> VAEEncode

    # Zone 4: Pass 2 Dual-Conditioning Prompt
    wb.connect(6, 0, 21, 0, "STRING")   # Clean Subject -> Concat Pass 2 (A)
    wb.connect(15, 2, 21, 1, "STRING")  # StudioLightRig lighting_prompt -> Concat Pass 2 (B)
    wb.connect(21, 0, 22, 0, "STRING")  # Concat Pass 2 -> PreviewAny
    wb.connect(21, 0, 23, 1, "STRING")  # Concat Pass 2 -> CLIPTextEncode
    wb.connect(23, 0, 24, 0, "CONDITIONING") # Pass 2 CLIPText -> FluxGuidance

    # Zone 5: Pass 2 Stochastic Re-Diffusion Sampler
    wb.connect(1, 0, 25, 0, "MODEL")    # UNET -> Pass 2 Sampler
    wb.connect(24, 0, 25, 1, "CONDITIONING") # Pass 2 Pos -> Pass 2 Sampler
    wb.connect(5, 0, 25, 2, "CONDITIONING")  # Neg -> Pass 2 Sampler
    wb.connect(20, 0, 25, 3, "LATENT")  # VAE Encoded Lit Plate -> Pass 2 Sampler

    # Final Decode & Save
    wb.connect(25, 0, 26, 0, "LATENT")  # Final Latent -> VAEDecode
    wb.connect(3, 0, 26, 1, "VAE")      # VAE -> VAEDecode
    wb.connect(26, 0, 27, 0, "IMAGE")   # Final Image -> SaveImage
    wb.connect(26, 0, 28, 0, "IMAGE")   # Final Image -> PreviewImage

    # Visual Zones / Groups
    groups = [
        {"title": "ZONE 1: FLUX LOADERS & PROMPT SEPARATION", "bounding": [-1420, -320, 720, 700], "color": "#1f2937"},
        {"title": "ZONE 2: PASS 1 DRAFT GENERATION (STEPS 0 -> 10)", "bounding": [-700, -320, 360, 840], "color": "#1e3a8a"},
        {"title": "ZONE 3: 3D NORMALS & LINEAR PHOTOMETRIC COMPOSITING", "bounding": [-340, -320, 720, 1020], "color": "#065f46"},
        {"title": "ZONE 4: DUAL-CONDITIONING PROMPT SYNTHESIS", "bounding": [380, -320, 340, 530], "color": "#4c1d95"},
        {"title": "ZONE 5: PASS 2 STOCHASTIC RE-DIFFUSION (THE CALIBRATION LEVER)", "bounding": [720, -320, 780, 720], "color": "#831843"},
    ]

    return wb.build(groups)

if __name__ == "__main__":
    wf = build_method_a()
    out_path = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_MethodA_Calibrated.json"
    with open(out_path, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"Successfully generated {out_path} with {len(wf['nodes'])} nodes and {len(wf['links'])} links.")
