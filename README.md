# ComfyUI-StudioLight

> **Professional Photographic & Fashion 3-Point Studio Lighting Controller for ComfyUI.**  
> Stop fighting prompt RNG for studio lighting. Shape Key, Fill, and Rim lights with real photographic angles, Kelvin color temperatures, optical modifiers, live 2D radar control, and latent-level physical illumination transfer.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Optimized for: Flux.1 Dev & SDXL](https://img.shields.io/badge/Optimized%20for-Flux.1%20%7C%20SDXL-black?style=flat-square)
![Phase 0.5: Passed](https://img.shields.io/badge/Phase%200.5%20Lighting%20Transfer-Verified-brightgreen?style=flat-square)

---

## The Philosophy

> **"Pose the geometry. Place the lights. Generate the image."**  
> Prompting lighting with text alone (*"dramatic rim light, soft key"*) is unreliable diffusion RNG. Words bleed into anatomy and cannot guarantee where photons strike. StudioLight bridges real-world photographic studio physics into generative diffusion models.

---

## ⚡ Current Status: Shipped in v0.1

### 1. `StudioLightRig` (`nodes/studio_light.py`)
Primary 3-point photographic lighting console with:
* **Interactive 2D Studio Radar:** Visual top-down radar on the node face with click-and-drag angle adjustment.
* **Photometric Shader Engine:** 32-bit Half-Lambert diffuse wrap + Blinn-Phong specular shading over surface normals.
* **Photographic Modifiers:** Octabox (150cm), Silver Beauty Dish, Fresnel Spotlight, Hard Snoot, Stripbox, and Direct On-Camera Flash.
* **Kelvin Color Science:** Realistic blackbody radiation curves ($2500\text{K} - 8000\text{K}$).
* **Dual VFX Calibration Probes:** Instant 0.03s generation of an 18% Neutral Grey Ball + Mirror Chrome Ball + 2D Floorplan Diagram.
* **Curated Studio Presets:** `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.

### 2. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
Tonal pre-processor for initial latent lighting transfer:
* **`shadow_amount`**: Independent shadow depth control (`-0.20` to crush shadows into deep, rich blacks).
* **`highlight_amount`**: Independent specular highlight boost (`+0.30` to punch up rim light reflections).
* **`shadow_tonal_width` / `highlight_tonal_width`**: Adjusts tonal reach into midtones.
* **`midtone_contrast`**: Fine midtone punch adjustment without blowing out highlights.

---

## 🔬 Empirical Discovery: The Phase 0.5 Lighting Benchmark

Before building the frontend 3D viewport, we stress-tested 10 different lighting transfer representations in Flux Dev / Krea on an NVIDIA RTX 4090 to determine how diffusion models best receive physical illumination:

| Representation Tested | Pose Lock | Light Direction Match | Texture Realism | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Pure Lit Mannequin (`denoise 0.85`)** | **100%** | **High** | **Photorealistic 8K Skin** | 🏆 **Winning Core Standard** |
| **Pure Mannequin (`0.90`) + Low Redux (`0.15`)** | **100%** | **Highest** | **Photorealistic 8K Skin** | 🥇 **Precision Hybrid Stack** |
| **Contrast-Tuned Mannequin + Initial Latent** | **100%** | **High** | **Punchy Speculars** | 🌟 **Production Favorite** |
| **SI Shadows & Highlights Pre-Processing** | **100%** | **High** | **Controlled Shadows** | 🟢 **Shipped in v0.1** |
| Gaussian Blurred Mannequin (`0.65 - 0.75`) | 80% | Moderate | Blurry Halo Artifacts | ❌ Failed (Halo light bleed) |
| Redux + Empty Latent (`1.0`) | 30% | Weak | High Realism | ❌ Failed (Random pose drift) |
| Blurred Image into Redux | 60% | Moderate | Forced Baldness / Buzzcut | ❌ Failed (SigCLIP bald token) |
| ControlNet Union (Lighting Mode) | 0% | None | Prompt-Driven Only | ❌ Failed (ControlNet is blind) |

### Key Takeaways:
1. **ControlNet Union is Blind to Light Maps:** Standard ControlNet Union models were never trained on illumination gradients and fail to steer light direction.
2. **Composite Signal via Initial Latents Succeeds:** Passing the lit mannequin plate into Flux's initial VAE latent at **`denoise 0.85`** strips plastic clay textures while preserving low-frequency directional luminance and 3-point contrast ratios.
3. **Contrast Tuning Acts as an Illumination Volume Knob:** Pre-processing the proxy with contrast adjustment or `SI Shadows & Highlights` gives Flux clear specular and shadow vectors to lock onto.

---

## 🗺️ Product Roadmap: The v1.0 3D Studio (In Active Development)

The upcoming v1.0 release introduces an embedded WebGL 3D Studio directly inside ComfyUI:

```text
                                  LIGHTING NODE (v1.0)
                                           │
           ┌───────────────────────────────┼───────────────────────────────┐
           │                               │                               │
  ① CREATE / IMPORT SCENE         ② LIGHT THE SCENE               ③ GENERATE & OUTPUT
   • 3D VRM Mannequin (FK Posing)  • 3D Key / Fill / Rim Gizmos    • Flux / Krea Pipeline
   • 3D Asset Upload (.glb / .obj) • Modifier & Kelvin Control     • Initial Latent (Denoise 0.85)
   • Match Pose From Image         • WebGL Native Normal/Depth Bake • Photorealistic Studio Render
```

* **Mode 1: 3D VRM Mannequin:** Forward Kinematics (FK) joint rotation gizmos on head, spine, shoulders, arms, and legs with 8 one-click fashion presets.
* **Mode 2: 3D Asset Upload (`.glb` / `.gltf` / `.obj`):** Universal 3D lighting for products, perfume bottles, footwear, vehicles, and sculptures.
* **Mode 3: Match Pose From Image:** Upload any reference photo $\to$ AI keypoint extraction maps joints to the VRM skeleton $\to$ loads an editable 3D pose in the studio.

---

## Architecture & Wiring (v0.1)

```text
                  ┌────────────────────────────────────────────────────────┐
                  │                 StudioLightRig                         │
                  ├────────────────────────────────────────────────────────┤
 (IMAGE, opt) ──► │ [●] normals_optional                                   │
 (IMAGE, opt) ──► │ [●] depth_optional                                     │
 (CLIP, opt)  ──► │ [●] clip_optional                                      │
 (COND, opt)  ──► │ [●] conditioning_in (from PromptBlock)                 │
                  │                                                        │
                  │   Preset: [ Rembrandt Classic                      ▼ ] │
                  │                                                        │
                  │   [●] Interactive 2D Studio Radar (Live Canvas)        │
                  │                                                        │
                  │   KEY LIGHT:  +45° Azimuth | 35° Elev | 5500K | Octa   │
                  │   FILL LIGHT: -30° Azimuth | 15° Elev | 4800K | Octa   │
                  │   RIM LIGHT: +160° Azimuth | 45° Elev | 6500K | Strip  │
                  ├────────────────────────────────────────────────────────┤
                  │ [●] relighting_map ────► [SI Shadows & Highlights]     │
                  │ [●] probe_preview  ────► [PreviewImage] (Instant VFX)  │
                  │ [●] conditioning   ────► [FluxGuidance / KSampler]     │
                  │ [●] lighting_prompt ───► [ShowText / CLIPTextEncode]   │
                  └────────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                     [SI Shadows & Highlights]
                                               │
                                               ▼
                                         [VAE Encode]
                                               │
                                               ▼
                                  [KSampler (Denoise: 0.85)]
                                               │
                                               ▼
                                   🏆 Photorealistic Output
```

---

## Installation

Inside your ComfyUI directory:

```bash
cd custom_nodes
git clone https://github.com/maciejtajber/ComfyUI-StudioLight.git
```

Restart ComfyUI. Find the nodes under:  
* **`StudioLight` → `Studio Light Rig (3-Point Photographic Console)`**
* **`StudioLight/processing` → `SI Shadows & Highlights`**
