# ComfyUI-StudioLight

> **Professional Photographic 3-Point Studio Lighting Console & Photometric Engine for ComfyUI.**  
> Direct the physical illumination of generative diffusion models with spatial control. Position Key, Fill, and Rim lights in 3D camera-space, dial blackbody-inspired Kelvin temperatures, shape beam angles with optical modifiers, control photographic distance falloff, inspect live 2D radar gizmos, and composite with scene-referred linear photometric math.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Calibrated for: Flux.1 Dev | Krea | Wan 2.1 | SDXL](https://img.shields.io/badge/Calibrated%20for-Flux.1%20%7C%20Krea%20%7C%20Wan%202.1%20%7C%20SDXL-black?style=flat-square)
![Automated Suites: 16/16 Verified](https://img.shields.io/badge/Automated%20Suites-16%2F16%20Verified-brightgreen?style=flat-square)
![Status: Calibrated Prototype](https://img.shields.io/badge/Status-Calibrated%20Prototype-blue?style=flat-square)

---

## The Concept

> **"Photoshop manipulates pixels. Generative models infer lighting from learned visual priors. StudioLight lets you design the lighting spatially in a virtual studio."**  

In standard generative AI, typing `"dramatic golden hour rim light"` or `"3200K tungsten spotlight"` is prompt gambling—text tokens bleed into anatomy, scramble facial features, and provide zero deterministic spatial control over where illumination strikes or casts shadows.

**How ComfyUI-StudioLight bridges photographic studio lighting into generative diffusion:**

1. **Midstream Structural Extraction:**  
   At an empirically calibrated intermediate step (e.g. step 10), the generative model has crystallized structural scene geometry, depth, and spatial forms. StudioLight extracts surface normals (DSINE) and metric depth (Depth Anything V2) from this latent draft and uses them as a lighting carrier.
2. **3D Camera-Space Back-Projection ($40^\circ$ Portrait FOV):**  
   Unprojects extracted 2D normals and depth into metric OpenGL camera-space ($Z = -1.5\text{m}$ to $-4.5\text{m}$), fully corrected for non-square anamorphic aspect ratios ($W/H$).
3. **Optical Penumbra & Area Soft Shadows (Screen-Space PCSS):**  
   Simulates physical modifier aperture diameters (Octabox 140cm, Beauty Dish 55cm, Snoot 10cm, Flash 4cm). Uses Percentage-Closer Soft Shadows with 8-tap Vogel disk filtering to keep contact shadows razor-sharp near occluders while softly diffusing cast shadows across distant walls.
4. **Calibrated Photographic Power Falloff ($1/d^{1.6}$) & Hermite Distance Taper:**  
   Calculates per-pixel Euclidean distance from the virtual lighting fixture to every surface point. A calibrated $1/d^{1.6}$ falloff curve is used to approximate practical studio light behavior and secondary bounce in enclosed volumes, while a smooth Hermite background taper ($d_{\text{norm}} < 0.20$) prevents studio strobes from spilling onto distant architecture 25 meters behind the subject.
5. **Optical Forward Modifier Beam Cones & Horizon Gating:**  
   Studio fixtures (Stripboxes, Beauty Dishes, Snoots) are aimed forward at the subject. The engine models the forward beam cone ($\cos\theta_{\text{beam}} > 0$) and specular horizon gating ($N \cdot L > 0$), eliminating backward light leakage and banishing edge halos on background architecture.
6. **Dedicated Specular Lobes & Rational Shoulder Tonemapping:**  
   Isolates pure Blinn-Phong dielectric catchlight lobes (tight 2.8% pixel area) composited onto linear Rec.709 plates, rolled off smoothly through a $C^1$-continuous rational shoulder with active `white_point` headroom control.
7. **Albedo Prompt Decoupling:**  
   Addresses the pre-baked specular artifact problem. Prompts are neutralized for Pass 1 with shadowless albedo tokens to generate clean, un-burned base geometry, allowing 3D specular catchlights in Pass 2 to appear strictly where your lights strike.
8. **Production Architecture (Method A):**  
   2-Stage Calibrated Relighting. Step 10 draft $\to$ 3D extraction $\to$ deferred photometric composite $\to$ stochastic re-diffusion (Steps 10–30, denoise `0.50–0.75`, empirically calibrated per subject/material). *(Continuous trajectory injection Method B is tracked for next-generation research in [ROADMAP.md](ROADMAP.md)).*

---

## 📦 The Core Production Suite (Method A)

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              METHOD A PRODUCTION PIPELINE                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

   ┌───────────────────────┐                                                              
   │  Master User Prompt   │                                                              
   └──────────┬────────────┘                                                              
              │                                                                           
              ▼                                                                           
   ┌───────────────────────┐                                                              
   │  Albedo Decoupling    │ ─── Pass 1 Albedo Prompt ───► ┌───────────────────────────┐  
   │ (Shadowless Carrier)  │                               │  Pass 1 Draft (Step 0-10) │  
   └──────────┬────────────┘                               └─────────────┬─────────────┘  
              │                                                          │ (Step 10 Latent)
              │ Pass 2 Full Prompt                                       ▼                
              │                                            ┌───────────────────────────┐  
              │                                            │  Extract 3D Normals/Depth │  
              │                                            │  (DSINE + DepthAnything)  │  
              │                                            └─────────────┬─────────────┘  
              │                                                          │                
              │                                                          ▼                
              │         ┌─────────────────────────┐        ┌───────────────────────────┐  
              │         │     StudioLightRig      │ ◄───── │   Extracted 3D Geometry   │  
              │         │(3-Point Lighting Console│        └───────────────────────────┘  
              │         └───────────┬─────────────┘                                       
              │                     │ (relighting_map + specular_pass)                    
              │                     ▼                                                     
              │         ┌─────────────────────────┐                                       
              │         │StudioPhotometricComposite│                                      
              │         │(Scene-Referred Linear)  │                                       
              │         └───────────┬─────────────┘                                       
              │                     │ (Lit Proxy Plate)                                   
              │                     ▼                                                     
              │         ┌─────────────────────────┐                                       
              │         │   SIShadowHighlight     │                                       
              │         │(Tonal Latent Sculptor)  │                                       
              │         └───────────┬─────────────┘                                       
              │                     │                                                     
              ▼                     ▼                                                     
   ┌──────────────────────────────────────────────┐                                       
   │            Pass 2 KSampler (Step 10-30)      │                                       
   │      (Photometric Stochastic Polish)         │ ───► 💎 FINAL MASTER RENDER           
   └──────────────────────────────────────────────┘                                       
```

### 1. `StudioLightRig` (`nodes/studio_light.py`)
The master 3-point photographic lighting console:
* **Interactive 2D Studio Radar**: Visual top-down radar on the node canvas with click-and-drag angle adjustment.
* **3-Point Studio Control**: Independent Azimuth ($-180^\circ$ to $+180^\circ$), Elevation ($-30^\circ$ to $+90^\circ$), Stand Distance ($0.5\text{m}$ to $10.0\text{m}$), and Intensity ($0.0$ to $3.0$) for Key, Fill, and Rim lights.
* **Photographic Modifiers**: Octabox (150cm soft wrap), Silver Beauty Dish, Fresnel Spotlight, Hard Snoot, Stripbox, and Direct On-Camera Flash.
* **Blackbody-Inspired Kelvin**: Blackbody-inspired Kelvin-to-RGB conversion with linear-light Rec.709 compositing ($2000\text{K} - 10000\text{K}$).
* **Dual-Sphere VFX Preview**: Real-time 18% Grey Ball + Mirror Chrome Ball + Studio Radar card rendered in 0.03s.
* **Curated Studio Presets**: `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.
* **4 Dedicated Outputs**: Exposes `relighting_map` (32-bit deferred plate), `specular_pass` (isolated Blinn-Phong lobes), `probe_preview`, and synchronized `lighting_prompt`.
* **Coordinate Parity Toggles**: `invert_normals_x` (DSINE / OpenGL parity), `invert_normals_y`, and `invert_depth` (near=white vs near=black polarity).

### 2. `StudioPhotometricComposite` (`nodes/photometric_composite.py`)
Scene-referred linear Rec.709 photographic compositing engine:
* **Zero Double-Floor Architecture**: Operates strictly in linear radiance space without compounding ambient floors.
* **Rational Shoulder Tonemapping**: Smooth $C^1$-continuous rational shoulder that compresses high-energy specular highlights into display headroom without hard-clipping.
* **Active `white_point` Control**: Dial highlight dynamic range compression stiffness from 1.5 (tight ceiling) to 8.0 (extended filmic headroom).
* **Lobe-Accurate Specular Integration**: Accepts isolated `specular_pass` input, confining dielectric catchlights to genuine specular microfacets instead of veiling entire lit surfaces.
* **Chroma Bleed Control**: Transfers Kelvin color warmth into midtones without contaminating deep shadows or dark fabrics.

### 3. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
Latent-level digital darkroom tonal sculptor:
* **`shadow_amount`**: Deepens shadows into rich noir blacks ($-0.30$) or lifts dark fabrics ($+0.25$).
* **`highlight_amount`**: Boosts specular catchlights on jewelry, metallic chains, and glossy leather.
* **`shadow_tonal_width` / `highlight_tonal_width`**: Shapes the transition zone into midtones.
* **`midtone_contrast`**: Tunes tonal punch without blowing out highlight detail.

### 4. `StudioPromptSanitizer` (`nodes/prompt_sanitizer.py`)
Automatic albedo decoupling engine:
* Strips high-gloss/specular terms from Pass 1 while delivering the full user prompt to Pass 2.
* Mitigates pre-baked specular blobs on patent leather, glass, and wet surfaces before they bake into the latent carrier.

---

## 🔬 Empirical Discoveries & Working Rules

Through empirical calibration on FLUX Krea Dev across 30+ iterations, we observed 4 working rules for practical generative studio lighting:

### Rule 1: Pass 1 Functions Best as a Flat Albedo Carrier
* **Observation:** If Pass 1 is prompted with directional sunlight or extreme contrast, the diffusion model bakes hard shadow contours into the geometry. Pass 2 may then sharpen that pre-baked edge into an unnatural seam.
* **Approach:** Use an Albedo Carrier Suffix in Pass 1 (`", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog"`). Pass 1 generates clean, unlit geometry, giving `StudioLightRig` cleaner spatial control over illumination gradients.

### Rule 2: Matte Fabrics Require Dedicated Specular Isolation
* **Observation:** Dark wool coats have an albedo near $0.02$. High specular kicks ($> 2.0$) with low ambient floors cause grazing Fresnel spikes, making black wool look like metallic foil or thermal imaging.
* **Approach:** Wire `specular_pass` directly into `StudioPhotometricComposite`, keep `specular_kick` modest ($0.30 – 0.50$), and set `chroma_bleed = 0.00` on dark garments. Direct diffuse radiance sculpts the form without solarizing matte wool fibers.

### Rule 3: Rear Lights Benefit from Forward Beam Coning & Distance Tapers
* **Observation:** Point lights placed behind the subject illuminate background architecture behind the light stand at $160^\circ+$ off-axis, creating edge halos on arches and washing out distant walls.
* **Approach:** The optical forward modifier beam cone ($\cos\theta_{\text{beam}} > 0$) models directional fixture housing, and the Hermite distance taper ($d_{\text{norm}} < 0.20$) drops strobes to 0% irradiance on distant geometry.

### Rule 4: Pass 2 Denoise Governs Illumination Integration
* **`0.35 – 0.45` (Feature Lock / Soft Commercial):** Strict geometric and facial lock. Best suited for soft catalog fill (Beauty Dish / Octabox).
* **`0.50 – 0.75` (Calibrated Range):** The working zone for dramatic lighting and chiaroscuro. Generatively synthesizes subsurface scattering on skin, new eye catchlights, and leather reflections while preserving human likeness. Denoise is empirically calibrated per subject and material.
* **`0.75+` (Hallucination Boundary):** At very high denoise, dark proxy corners combined with prompt tokens like `"Stripbox"` can cause the model to hallucinate physical studio softboxes hanging in the frame. Keep denoise $\le 0.75$.

---

## 🚀 Quick Start & Installation

### Option 1: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maciejtajber/ComfyUI-StudioLight.git
pip install -r ComfyUI-StudioLight/requirements.txt
```

### Option 2: ComfyUI Manager
Search for `ComfyUI-StudioLight` in the ComfyUI Manager and click Install.

### Running Automated Test Suite
To verify math, color conversions, shader engines, and node schemas:
```bash
python test_studiolight.py
```
*(Runs 100% standalone anywhere with PyTorch—zero external ComfyUI runtime required).*

---

## 📁 Included Master Workflows

Inside the `workflows/` directory:
* **`StudioLight_MethodA_Calibrated.json`**: The calibrated production standard (37 nodes with visual engineering notes and organized zones). Features FLUX Krea Dev, step-10 3D extraction (DSINE + Depth Anything V2), dual-conditioning prompt synthesis, dedicated specular pass, and live preview cards.
* **`StudioLight_Midstream_Master.json`**: Midstream KSampler hook architecture for fine-grained Euler step interception.
* **`StudioLight_Midstream_Sweep.json`**: Automated parameter sweep across lighting angles.

---

## 📄 License
MIT License. Created by [Maciej Tajber](https://github.com/maciejtajber).
