"""
Build StudioLight Midstream Injection Sweep Workflow
Tests injecting Python-calculated lighting into Flux at different denoise/timestep stages:
- Pass 1: Flux text-to-image draft (steps 0 to 10)
- Intercept: Surface normal extraction -> StudioLightRig (Key/Fill/Rim) -> StudioPhotometricComposite -> SIShadowHighlight
- Pass 2: 4-Way Denoise Sweep (0.75, 0.55, 0.40, 0.25) to identify the optimal lighting bonding point
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


def generate_workflow():
    wb = WorkflowBuilder()

    # --- ZONE 1: MODEL & TEXT PIPELINE (Left) ---
    # 1. UNETLoader
    wb.add_node(
        node_id=1,
        node_type="UNETLoader",
        title="Load Diffusion Model (Flux Dev)",
        pos=[-1200, -200],
        size=[320, 82],
        widgets_values=["flux1-krea-dev.safetensors", "default"],
        input_names=[],
        output_defs=[("MODEL", "MODEL")],
    )

    # 2. DualCLIPLoader
    wb.add_node(
        node_id=2,
        node_type="DualCLIPLoader",
        title="DualCLIPLoader (T5xxl + CLIP-L)",
        pos=[-1200, -80],
        size=[320, 106],
        widgets_values=["clip_l.safetensors", "t5xxl_fp16.safetensors", "flux", "default"],
        input_names=[],
        output_defs=[("CLIP", "CLIP")],
    )

    # 3. VAELoader
    wb.add_node(
        node_id=3,
        node_type="VAELoader",
        title="Load VAE (ae.safetensors)",
        pos=[-1200, 60],
        size=[320, 60],
        widgets_values=["ae.safetensors"],
        input_names=[],
        output_defs=[("VAE", "VAE")],
    )

    # 4. CLIPTextEncode (Positive Prompt with Flat Directive)
    wb.add_node(
        node_id=4,
        node_type="CLIPTextEncode",
        title="Subject Prompt (Flat Studio Lighting Directive)",
        pos=[-840, -200],
        size=[400, 180],
        widgets_values=[
            "A highly detailed mechanical rhinoceros beetle, iridescent metallic carapace with brass gears, resting on dark polished basalt stone, studio photography, sharp macro focus, flat high-key diffuse studio illumination, shadowless even ambient lighting"
        ],
        input_names=[("clip", "CLIP")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    # 5. FluxGuidance
    wb.add_node(
        node_id=5,
        node_type="FluxGuidance",
        title="FluxGuidance (3.5)",
        pos=[-840, 20],
        size=[220, 60],
        widgets_values=[3.5],
        input_names=[("conditioning", "CONDITIONING")],
        output_defs=[("CONDITIONING", "CONDITIONING")],
    )

    # 6. EmptyLatentImage (1024x1024)
    wb.add_node(
        node_id=6,
        node_type="EmptyLatentImage",
        title="Empty Latent (1024x1024)",
        pos=[-840, 120],
        size=[220, 106],
        widgets_values=[1024, 1024, 1],
        input_names=[],
        output_defs=[("LATENT", "LATENT")],
    )

    # --- ZONE 2: PASS 1 (THE DRAFT SAMPLER) ---
    # 7. KSamplerAdvanced (Draft to Step 10)
    wb.add_node(
        node_id=7,
        node_type="KSamplerAdvanced",
        title="Pass 1: Geometry Draft (Steps 0 -> 10)",
        pos=[-580, -200],
        size=[320, 320],
        widgets_values=[
            "enable",  # add_noise
            424242,    # seed
            "fixed",   # control_after_generate
            20,        # steps
            1.0,       # cfg
            "euler",   # sampler_name
            "simple",  # scheduler
            0,         # start_at_step
            10,        # end_at_step (Pause at step 10!)
            "disable", # return_with_leftover_noise
        ],
        input_names=[
            ("model", "MODEL"),
            ("positive", "CONDITIONING"),
            ("negative", "CONDITIONING"),
            ("latent_image", "LATENT"),
        ],
        output_defs=[("LATENT", "LATENT")],
    )

    # 8. VAEDecode (Draft Image)
    wb.add_node(
        node_id=8,
        node_type="VAEDecode",
        title="Decode Draft Image",
        pos=[-220, -200],
        size=[200, 60],
        widgets_values=[],
        input_names=[("samples", "LATENT"), ("vae", "VAE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    # 9. PreviewImage (Draft Image Preview)
    wb.add_node(
        node_id=9,
        node_type="PreviewImage",
        title="Preview: Flat Geometry Draft",
        pos=[-220, -100],
        size=[260, 280],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # --- ZONE 3: SURFACE NORMAL EXTRACTION & PYTHON LIGHTING ---
    # 10. DSINE-NormalMapPreprocessor (Extract Normals from Draft)
    wb.add_node(
        node_id=10,
        node_type="DSINE-NormalMapPreprocessor",
        title="Extract 3D Normals (DSINE)",
        pos=[80, -200],
        size=[280, 140],
        widgets_values=[0.05, 1024],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    # 11. StudioLightRig (Our Photographic Console)
    wb.add_node(
        node_id=11,
        node_type="StudioLightRig",
        title="StudioLightRig (3-Point Console)",
        pos=[390, -200],
        size=[340, 520],
        widgets_values=[
            "Rembrandt Classic",  # preset
            "Tabletop / Luxury (Perfume Bottle / Small Product)",  # framing
            0.04,                 # ambient_intensity
            False,                # invert_normals_y
            True, 45.0, 35.0, 1.20, 5500, "Octabox (Soft)",    # Key Light
            True, -30.0, 15.0, 0.35, 4800, "Octabox (Soft)",   # Fill Light
            True, 160.0, 45.0, 1.15, 6500, "Stripbox",          # Rim Light
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

    # 12. PreviewImage (VFX Balls & Probes)
    wb.add_node(
        node_id=12,
        node_type="PreviewImage",
        title="Preview: VFX Lighting Balls",
        pos=[390, 360],
        size=[340, 260],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # 13. StudioPhotometricComposite (Relighting Compositor)
    wb.add_node(
        node_id=13,
        node_type="StudioPhotometricComposite",
        title="Studio Photometric Composite (Linear HDR)",
        pos=[760, -200],
        size=[300, 220],
        widgets_values=[
            1.0,   # lighting_strength
            0.05,  # ambient_floor
            1.15,  # specular_kick
            0.40,  # chroma_bleed
            3.5,   # filmic_white_point
        ],
        input_names=[
            ("image", "IMAGE"),
            ("lighting_map", "IMAGE"),
        ],
        output_defs=[("IMAGE", "IMAGE")],
    )

    # 14. SIShadowHighlight (Volume & Specular Sculptor)
    wb.add_node(
        node_id=14,
        node_type="SIShadowHighlight",
        title="SI Shadows & Highlights",
        pos=[760, 60],
        size=[300, 180],
        widgets_values=[
            0.20,  # shadow_amount
            0.10,  # highlight_amount
            0.50,  # shadow_tonal_width
            0.50,  # highlight_tonal_width
            1.05,  # midtone_contrast
        ],
        input_names=[("image", "IMAGE")],
        output_defs=[("IMAGE", "IMAGE")],
    )

    # 15. PreviewImage (Lit Composite Plate Preview)
    wb.add_node(
        node_id=15,
        node_type="PreviewImage",
        title="Preview: Relit Subject Plate",
        pos=[760, 280],
        size=[300, 340],
        widgets_values=[],
        input_names=[("images", "IMAGE")],
        output_defs=[],
    )

    # 16. VAEEncode (Lit Plate -> Latent)
    wb.add_node(
        node_id=16,
        node_type="VAEEncode",
        title="VAE Encode Lit Plate",
        pos=[1090, -200],
        size=[200, 60],
        widgets_values=[],
        input_names=[("pixels", "IMAGE"), ("vae", "VAE")],
        output_defs=[("LATENT", "LATENT")],
    )

    # --- ZONE 4: THE 4-WAY DENOISE SWEEP ---
    sweep_configs = [
        ("Run A: Early Injection (Denoise 0.75)", 0.75, 1340, -200),
        ("Run B: Mid Injection (Denoise 0.55)", 0.55, 1340, 220),
        ("Run C: Late Injection (Denoise 0.40)", 0.40, 1340, 640),
        ("Run D: Ultra-Late Polish (Denoise 0.25)", 0.25, 1340, 1060),
    ]

    base_k_id = 20
    for idx, (title, denoise_val, px, py) in enumerate(sweep_configs):
        ks_id = base_k_id + idx * 3
        vd_id = ks_id + 1
        pv_id = ks_id + 2

        # KSampler
        wb.add_node(
            node_id=ks_id,
            node_type="KSampler",
            title=title,
            pos=[px, py],
            size=[300, 260],
            widgets_values=[
                424242,       # seed (identical seed for fair comparison!)
                "fixed",      # control_after_generate
                20,           # steps
                1.0,          # cfg
                "euler",      # sampler_name
                "simple",     # scheduler
                denoise_val,  # denoise sweep value!
            ],
            input_names=[
                ("model", "MODEL"),
                ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"),
                ("latent_image", "LATENT"),
            ],
            output_defs=[("LATENT", "LATENT")],
        )

        # VAEDecode
        wb.add_node(
            node_id=vd_id,
            node_type="VAEDecode",
            title=f"Decode: {title}",
            pos=[px + 330, py],
            size=[180, 60],
            widgets_values=[],
            input_names=[("samples", "LATENT"), ("vae", "VAE")],
            output_defs=[("IMAGE", "IMAGE")],
        )

        # PreviewImage
        wb.add_node(
            node_id=pv_id,
            node_type="PreviewImage",
            title=f"Output: {title}",
            pos=[px + 330, py + 80],
            size=[260, 280],
            widgets_values=[],
            input_names=[("images", "IMAGE")],
            output_defs=[],
        )

    # --- CONNECTIONS ---
    # Model & Conditioning wires
    wb.connect(2, 0, 4, 0, "CLIP")          # DualCLIP -> Prompt
    wb.connect(4, 0, 5, 0, "CONDITIONING")  # Prompt -> Guidance
    wb.connect(1, 0, 7, 0, "MODEL")         # UNET -> KSampler 1
    wb.connect(5, 0, 7, 1, "CONDITIONING")  # Guidance -> KSampler 1 positive
    wb.connect(6, 0, 7, 3, "LATENT")        # EmptyLatent -> KSampler 1

    # Pass 1 Decode & Normal extraction
    wb.connect(7, 0, 8, 0, "LATENT")        # KSampler 1 -> VAE Decode
    wb.connect(3, 0, 8, 1, "VAE")           # VAE -> VAE Decode
    wb.connect(8, 0, 9, 0, "IMAGE")         # VAE Decode -> Preview Draft
    wb.connect(8, 0, 10, 0, "IMAGE")        # VAE Decode -> DSINE Normals

    # Lighting Console & Compositing Pipeline
    wb.connect(10, 0, 11, 1, "IMAGE")       # DSINE Normals -> StudioLightRig normals_optional
    wb.connect(11, 1, 12, 0, "IMAGE")       # StudioLightRig probe_preview -> Preview VFX Balls
    wb.connect(8, 0, 13, 0, "IMAGE")        # Pass 1 Draft Image -> StudioPhotometricComposite image
    wb.connect(11, 0, 13, 1, "IMAGE")       # StudioLightRig relighting_map -> StudioPhotometricComposite lighting_map
    wb.connect(13, 0, 14, 0, "IMAGE")       # Lit Image -> SIShadowHighlight
    wb.connect(14, 0, 15, 0, "IMAGE")       # SIShadowHighlight -> Preview Relit Plate
    wb.connect(14, 0, 16, 0, "IMAGE")       # SIShadowHighlight -> VAE Encode pixels
    wb.connect(3, 0, 16, 1, "VAE")          # VAE -> VAE Encode

    # Connect the 4 Sweep Samplers
    for idx in range(4):
        ks_id = base_k_id + idx * 3
        vd_id = ks_id + 1
        pv_id = ks_id + 2

        wb.connect(1, 0, ks_id, 0, "MODEL")         # Model -> KSampler
        wb.connect(5, 0, ks_id, 1, "CONDITIONING")  # Positive -> KSampler
        wb.connect(16, 0, ks_id, 3, "LATENT")       # VAE Encode (Lit Plate) -> KSampler
        wb.connect(ks_id, 0, vd_id, 0, "LATENT")    # KSampler -> VAEDecode
        wb.connect(3, 0, vd_id, 1, "VAE")           # VAE -> VAEDecode
        wb.connect(vd_id, 0, pv_id, 0, "IMAGE")     # VAEDecode -> PreviewImage

    # Groups
    groups = [
        {"title": "ZONE 1: Model & Subject Prompt", "bounding": [-1240, -260, 640, 520], "color": "#232"},
        {"title": "ZONE 2: Pass 1 — Fast Geometry Draft (Steps 0->10)", "bounding": [-600, -260, 660, 480], "color": "#332"},
        {"title": "ZONE 3: 3D Surface Normals, Python Lighting & Photometric Composite", "bounding": [60, -260, 1250, 900], "color": "#324"},
        {"title": "ZONE 4: The 4-Way Denoise Sweep (0.75, 0.55, 0.40, 0.25)", "bounding": [1320, -260, 640, 1620], "color": "#243"},
    ]

    return wb.build(groups=groups)


if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    wf = generate_workflow()
    out_path = os.path.join(out_dir, "StudioLight_Midstream_Sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2)
    print(f"✓ StudioLight Midstream Sweep workflow generated successfully at:\n  {out_path}")
