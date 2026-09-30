"""
Builds StudioLight_Midstream_Master.json
Clean, production single-output Mid-Stream Latent Intercept Workflow.
- Pass 1: Flux text-to-image draft (10 steps)
- Intercept: DSINE Normals -> StudioLightRig -> StudioPhotometricComposite (Linear HDR) -> SIShadowHighlight
- Pass 2: Flux Photometric Polish (0.75 denoise - the proven winner!)
- Output: SaveImage (Midstream_FINAL_StudioLight) with giant preview
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


def generate_master_workflow():
    wb = WorkflowBuilder()

    # --- ZONE 1: MODEL & PROMPT PIPELINE ---
    wb.add_node(
        node_id=1,
        node_type="UNETLoader",
        title="Flux Krea Dev (Base UNET)",
        pos=[-1400, -200],
        size=[320, 82],
        widgets_values=["flux1-krea-dev.safetensors", "default"],
        input_names=[],
        output_defs=[("MODEL", "MODEL")],
    )

    wb.add_node(
        node_id=2,
        node_type="DualCLIPLoader",
        title="Dual CLIP Loader (T5xxl + CLIP-L)",
        pos=[-1400, -80],
        size=[320, 106],
        widgets_values=["clip_l.safetensors", "t5xxl_fp16.safetensors", "flux", "default"],
        input_names=[],
        output_defs=[("CLIP", "CLIP")],
    )

    wb.add_node(
        node_id=3,
        node_type="VAELoader",
        title="Flux VAE",
        pos=[-1400, 60],
        size=[320, 60],
        widgets_values=["ae.safetensors"],
        input_names=[],
        output_defs=[("VAE", "VAE")],
    )

    wb.add_node(
        node_id=4,
        node_type="CLIPTextEncode",
        title="Subject Prompt (Flat Studio Lighting Directive)",
        pos=[-1040, -200],
        size=[380, 180],
        widgets_values=[
            "A highly detailed mechanical rhinoceros beetle, iridescent metallic carapace with brass gears, resting on dark polished basalt stone, studio photography, sharp macro focus, flat high-key diffuse studio illumination, shadowless even ambient lighting"
        ],
        input_names=[("clip", "CLIP")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    wb.add_node(
        node_id=5,
        node_type="FluxGuidance",
        title="Flux Guidance (3.5)",
        pos=[-1040, 20],
        size=[220, 60],
        widgets_values=[3.5],
        input_names=[("conditioning", "CONDITIONING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    wb.add_node(
        node_id=6,
        node_type="EmptyLatentImage",
        title="Empty Latent (1024x1024)",
        pos=[-1040, 120],
        size=[220, 106],
        widgets_values=[1024, 1024, 1],
        input_names=[],
        output_defs=[("LATENT", "LATENT")],
    )

    # --- ZONE 2: PASS 1 (DRAFT GEOMETRY) ---
    wb.add_node(
        node_id=7,
        node_type="KSamplerAdvanced",
        title="Pass 1: Geometry Draft (Steps 0 -> 10)",
        pos=[-620, -200],
        size=[300, 320],
        widgets_values=[
            "enable",   # add_noise
            424242,     # noise_seed
            "randomize", # control_after_generate (RANDOMIZE BY DEFAULT!)
            20,         # steps
            1.0,        # cfg
            "euler",    # sampler_name
            "simple",   # scheduler
            0,          # start_at_step
            10,         # end_at_step (Pause at step 10!)
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
        node_id=8,
        node_type="VAEDecode",
        title="Decode Draft",
        pos=[-280, -200],
        size=[180, 60],
        widgets_values=[],
        input_names=[("samples", "LATENT"), ("vae", "VAE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=9,
        node_type="PreviewImage",
        title="⚠️ Pass 1 Draft (Steps 0->10 - Not Final)",
        pos=[-280, -100],
        size=[280, 300],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # --- ZONE 3: 3D NORMALS & STUDIO LIGHTING ---
    wb.add_node(
        node_id=10,
        node_type="DSINE-NormalMapPreprocessor",
        title="Extract 3D Normals (DSINE)",
        pos=[40, -200],
        size=[280, 140],
        widgets_values=[60.0, 5, 1024],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=11,
        node_type="StudioLightRig",
        title="StudioLightRig (3-Point Console)",
        pos=[360, -200],
        size=[340, 520],
        widgets_values=[
            "Rembrandt Classic",
            "Tabletop / Luxury (Perfume Bottle / Small Product)",
            0.04,
            False,
            True, 45.0, 35.0, 1.20, 5500, "Octabox (Soft)",
            True, -30.0, 15.0, 0.35, 4800, "Octabox (Soft)",
            True, 160.0, 45.0, 1.15, 6500, "Stripbox",
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
        node_id=12,
        node_type="PreviewImage",
        title="VFX Probes (18% Grey & Chrome + Radar)",
        pos=[360, 360],
        size=[340, 260],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    wb.add_node(
        node_id=13,
        node_type="StudioPhotometricComposite",
        title="Studio Photometric Composite (Linear HDR)",
        pos=[740, -200],
        size=[300, 220],
        widgets_values=[1.0, 0.05, 1.15, 0.40, 3.5],
        input_names=[
            ("image", "IMAGE"),
            ("lighting_map", "IMAGE"),
        ],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=14,
        node_type="SIShadowHighlight",
        title="SI Shadows & Highlights",
        pos=[740, 60],
        size=[300, 180],
        widgets_values=[0.20, 0.10, 0.50, 0.50, 1.05],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    wb.add_node(
        node_id=15,
        node_type="VAEEncode",
        title="VAE Encode Lit Plate",
        pos=[740, 280],
        size=[200, 60],
        widgets_values=[],
        input_names=[("pixels", "IMAGE"), ("vae", "VAE")],
        output_defs=[("LATENT", "LATENT")],
    )

    # --- ZONE 4: PASS 2 PHOTOMETRIC POLISH (THE PROVEN 28-STEP BETA FINISH) ---
    wb.add_node(
        node_id=16,
        node_type="KSampler",
        title="Pass 2: Photometric Finish (Denoise 0.72 - Beta)",
        pos=[1100, -200],
        size=[320, 280],
        widgets_values=[
            424242,
            "fixed",
            28,        # 28 STEPS FOR 100% PRODUCTION FINISH!
            1.0,
            "euler",
            "beta",    # BETA SCHEDULER!
            0.72,      # 0.72 OPTIMAL BONDING!
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
        node_id=17,
        node_type="VAEDecode",
        title="Decode Final Master",
        pos=[1460, -200],
        size=[200, 60],
        widgets_values=[],
        input_names=[("samples", "LATENT"), ("vae", "VAE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    # 18. SaveImage (PERMANENT FINAL OUTPUT!)
    wb.add_node(
        node_id=18,
        node_type="SaveImage",
        title="💎 FINAL LIT RENDER (Saved to Disk)",
        pos=[1460, -100],
        size=[440, 480],
        widgets_values=["StudioLight_Midstream_FINAL"],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # --- CONNECTIONS ---
    wb.connect(2, 0, 4, 0, "CLIP")          # DualCLIP -> Prompt
    wb.connect(4, 0, 5, 0, "CONDITIONING")  # Prompt -> Guidance
    wb.connect(1, 0, 7, 0, "MODEL")         # UNET -> KSampler 1
    wb.connect(5, 0, 7, 1, "CONDITIONING")  # Guidance -> KSampler 1 positive
    wb.connect(5, 0, 7, 2, "CONDITIONING")  # Guidance -> KSampler 1 negative
    wb.connect(6, 0, 7, 3, "LATENT")        # EmptyLatent -> KSampler 1

    wb.connect(7, 0, 8, 0, "LATENT")        # KSampler 1 -> VAE Decode
    wb.connect(3, 0, 8, 1, "VAE")           # VAE -> VAE Decode
    wb.connect(8, 0, 9, 0, "IMAGE")         # VAE Decode -> Preview Draft
    wb.connect(8, 0, 10, 0, "IMAGE")        # VAE Decode -> DSINE Normals

    wb.connect(10, 0, 11, 1, "IMAGE")       # DSINE Normals -> StudioLightRig normals_optional
    wb.connect(11, 1, 12, 0, "IMAGE")       # StudioLightRig probe_preview -> Preview VFX Balls
    wb.connect(8, 0, 13, 0, "IMAGE")        # Pass 1 Draft Image -> StudioPhotometricComposite image
    wb.connect(11, 0, 13, 1, "IMAGE")       # StudioLightRig relighting_map -> StudioPhotometricComposite lighting_map
    wb.connect(13, 0, 14, 0, "IMAGE")       # Lit Image -> SIShadowHighlight
    wb.connect(14, 0, 15, 0, "IMAGE")       # SIShadowHighlight -> VAE Encode pixels
    wb.connect(3, 0, 15, 1, "VAE")          # VAE -> VAE Encode

    wb.connect(1, 0, 16, 0, "MODEL")         # UNET -> Pass 2 KSampler
    wb.connect(5, 0, 16, 1, "CONDITIONING")  # Guidance -> Pass 2 KSampler positive
    wb.connect(5, 0, 16, 2, "CONDITIONING")  # Guidance -> Pass 2 KSampler negative
    wb.connect(15, 0, 16, 3, "LATENT")       # VAE Encode (Lit Plate) -> Pass 2 KSampler
    wb.connect(16, 0, 17, 0, "LATENT")       # Pass 2 KSampler -> VAEDecode
    wb.connect(3, 0, 17, 1, "VAE")           # VAE -> VAEDecode
    wb.connect(17, 0, 18, 0, "IMAGE")        # VAEDecode -> SaveImage (FINAL OUTPUT!)

    # Groups
    groups = [
        {"title": "ZONE 1: Model Stack & Subject Prompt", "bounding": [-1440, -260, 780, 520], "color": "#232"},
        {"title": "ZONE 2: Pass 1 — Fast Geometry Draft (Steps 0->10)", "bounding": [-640, -260, 660, 480], "color": "#332"},
        {"title": "ZONE 3: 3D Surface Normals, Python Lighting & Photometric Composite", "bounding": [20, -260, 1040, 900], "color": "#324"},
        {"title": "ZONE 4: Pass 2 — Photometric Polish & FINAL OUTPUT", "bounding": [1080, -260, 840, 700], "color": "#243"},
    ]

    return wb.build(groups=groups)


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    wf = generate_master_workflow()
    out_path = os.path.join(out_dir, "StudioLight_Midstream_Master.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    print(f"✓ StudioLight Midstream Master workflow generated successfully at:\n  {out_path}")
