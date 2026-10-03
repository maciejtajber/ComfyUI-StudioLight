# ComfyUI-StudioLight

> **Professional Photographic 3-Point Studio Lighting Console & Virtual Photon Engine for ComfyUI.**  
> Direct the physical illumination of generative diffusion models with mathematical precision. Position Key, Fill, and Rim lights in 3D camera-space, dial real-world blackbody Kelvin temperatures, shape beam angles with optical modifiers, control inverse-square falloff, inspect live 2D radar gizmos, and composite with scene-referred linear photometric physics.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Optimized for: Flux.1 Dev | Flux.2 | SDXL | Video](https://img.shields.io/badge/Optimized%20for-Flux.1%20%7C%20Flux.2%20%7C%20SDXL%20%7C%20Wan2.1-black?style=flat-square)
![Tests: 15/15 Passed](https://img.shields.io/badge/Automated%20Suites-15%2F15%20Verified-brightgreen?style=flat-square)
![Status: Production Calibrated](https://img.shields.io/badge/Status-Production%20Calibrated-success?style=flat-square)

---

## The Breakthrough

> **"Photoshop paints 2D pixels. Generative AI hallucinates lighting from text prompts. StudioLight simulates 3D virtual photons."**  

In standard generative AI, typing `"dramatic golden hour rim light"` or `"3200K tungsten spotlight"` is pure prompt gambling—text tokens bleed into anatomy, scramble facial features, and provide zero physical control over where photons strike or cast shadows.

**ComfyUI-StudioLight bridges professional studio lighting physics directly into generative diffusion models:**

1. **Midstream 3D Extraction (The Step 10 Law):**  
   We do **not** rely on external 3D geometry or manual DCC blockouts. At step 10 of the diffusion trajectory, the generative model has crystallized structural scene geometry, depth, and spatial forms. We extract high-fidelity 3D surface normals (DSINE) and metric depth (Depth Anything V2) directly from this step-10 latent draft.
2. **3D Camera-Space Back-Projection ($40^\circ$ Portrait FOV):**  
   Unprojects extracted 2D normals and depth into metric OpenGL camera-space ($Z = -1.5\text{m}$ to $-4.5\text{m}$), fully corrected for non-square anamorphic aspect ratios ($W/H$).
3. **True Inverse-Square Power Falloff ($1/d^{1.6}$) & Hermite Distance Taper:**  
   Calculates true per-pixel Euclidean distance from the virtual lighting fixture to every surface point. A close key light ($0.8\text{m}$) casts high-contrast, steep inverse-square falloff onto foreground geometry, while a smooth Hermite background taper ($d_{\text{norm}} < 0.20$) prevents studio strobes from spilling onto distant architecture 25 meters behind the subject.
4. **Optical Forward Modifier Beam Cones:**  
   Studio fixtures (Stripboxes, Beauty Dishes, Snoots) are aimed forward at the subject. The engine models the physical forward beam cone ($\cos\theta_{\text{beam}} > 0$), eliminating backward light leakage and banishing silhouette edge halos.
5. **Dedicated Specular Lobes & Rational Shoulder Tonemapping:**  
   Isolates pure Blinn-Phong dielectric catchlight lobes (2.8% pixel area) composited onto linear Rec.709 plates, rolled off smoothly through a $C^1$-continuous rational shoulder with active `white_point` headroom control.
6. **Automatic Albedo Prompt Decoupling (`StudioPromptSanitizer`):**  
   Solves the pre-baked specular artifact problem. Users write normal descriptive prompts (e.g. `"glossy black patent leather handbag"`). The engine automatically filters high-specular tokens for Pass 1 (Steps 0–10) to obtain clean, un-burned albedo, then restores the full prompt for Pass 2 so 3D specular catchlights appear strictly where your lights strike.
7. **Two Battle-Tested Architectures:**
   * **Method A (2-Stage Calibrated Relighting):** The production workhorse. Step 10 draft $\to$ 3D extraction $\to$ deferred photometric composite $\to$ stochastic re-diffusion (Steps 10–30).
   * **Method B (Continuous Trajectory Injection):** Mathematical flow-matching split at flow step $\sigma_{10} = 0.863338$, intercepting clean $x_0$ via native Euler callback and algebraically injecting the lighting delta $\Delta x = (x_0' - x_0) \cdot \text{strength}$ directly into the diffusion ODE path.

---

## 📦 The Core Production Suite

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               METHOD A PRODUCTION PIPELINE                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                          
   ┌───────────────────────┐                                                              
   │  Master User Prompt   │                                                              
   └──────────┬────────────┘                                                              
              │                                                                           
              ▼                                                                           
   ┌───────────────────────┐                                                              
   │ StudioPromptSanitizer │ ─── Pass 1 Albedo Prompt ───► ┌───────────────────────────┐  
   │ (Auto-Decouple Gloss) │                               │  Pass 1 Draft (Step 0-10) │  
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
              │                     │ (Relighting Map + Specular Pass)                    
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
* **Interactive 2D Studio Radar:** Visual top-down radar on the node canvas with click-and-drag angle adjustment.
* **3-Point Studio Control:** Independent Azimuth ($-180^\circ$ to $+180^\circ$), Elevation ($-30^\circ$ to $+90^\circ$), Stand Distance ($0.5\text{m}$ to $10.0\text{m}$), and Intensity ($0.0$ to $3.0$) for Key, Fill, and Rim lights.
* **Photographic Modifiers:** Octabox (150cm soft wrap), Silver Beauty Dish, Fresnel Spotlight, Hard Snoot, Stripbox, and Direct On-Camera Flash.
* **Blackbody Radiation Science:** Calibrated Kelvin curves ($2000\text{K} - 10000\text{K}$) mapped to IEC 61966-2-1 linear Rec.709 spectral radiance.
* **Dual-Sphere VFX Preview:** Real-time 18% Grey Ball + Mirror Chrome Ball + Studio Radar card rendered in 0.03s.
* **Curated Studio Presets:** `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.
* **Dedicated Outputs:** Exposes `relighting_map` (32-bit deferred plate), `specular_pass` (isolated Blinn-Phong lobes), `probe_preview`, and synchronized `lighting_prompt`.
* **Coordinate Parity Toggles:** `invert_normals_x` (DSINE / OpenGL parity), `invert_normals_y`, and `invert_depth` (near=white vs near=black polarity).

### 2. `StudioPhotometricComposite` (`nodes/photometric_composite.py`)
Scene-referred linear Rec.709 photographic compositing engine:
* **Zero Double-Floor Architecture:** Operates strictly in linear radiance space without compounding ambient floors.
* **Rational Shoulder Tonemapping:** Smooth $C^1$-continuous rational shoulder that compresses high-energy specular highlights into display headroom without hard-clipping.
* **Active `white_point` Control:** Dial highlight dynamic range compression stiffness from 1.5 (tight ceiling) to 8.0 (extended filmic headroom).
* **Lobe-Accurate Specular Integration:** Accepts isolated `specular_pass` input, confining dielectric catchlights to genuine specular microfacets instead of veiling entire lit surfaces.
* **Chroma Bleed Control:** Transfers Kelvin color warmth into midtones without contaminating deep shadows or dark fabrics.

### 3. `StudioPromptSanitizer` (`nodes/prompt_sanitizer.py`)
Automatic albedo decoupling engine:
* **Zero User Friction:** The user inputs their normal, natural master prompt with full creative freedom.
* **Albedo Neutralization:** Automatically sanitizes high-specular terms (`"glossy"`, `"patent"`, `"shiny"`, `"chrome"`, `"reflective"`) for Pass 1 (Steps 0–10) and appends a shadowless catalog albedo suffix.
* **Full Fidelity Pass 2:** Delivers the complete master prompt to Pass 2 so FLUX materializes rich gloss, liquid sheen, and dielectric catchlights exactly where your 3D lights strike.

### 4. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
Latent-level digital darkroom tonal sculptor:
* **`shadow_amount`**: Deepens shadows into rich noir blacks (`-0.30`) or lifts dark fabrics (`+0.25`).
* **`highlight_amount`**: Boosts specular catchlights on jewelry, metallic chains, and glossy leather.
* **`shadow_tonal_width` / `highlight_tonal_width`**: Shapes the transition zone into midtones.
* **`midtone_contrast`**: Tunes tonal punch without blowing out highlight detail.

### 5. `StudioLightTrajectory` Nodes (`nodes/trajectory_nodes.py`)
Method B continuous diffusion flow injection:
* **`StudioLightTrajectorySplitter`**: Executes diffusion from step 0 to the split step (calibrated at $\sigma_{10} = 0.863338$), intercepting un-overshot clean prediction $x_0$ via native Euler callback.
* **`StudioLightTrajectoryInject`**: Combines relit latent $x_0'$ with noisy ODE state $x_t$ via $\text{injected} = x_t + (x_0' - x_0) \cdot \text{strength}$.

---

## 🔬 Empirical Discoveries & Forensic Rules

Through exhaustive empirical calibration on FLUX Krea Dev across 30+ iterations, we established 4 fundamental laws of generative lighting:

### Law 1: Pass 1 Must Be a Flat Albedo Carrier
* **The Problem:** If Pass 1 is prompted with directional sunlight or contrast, FLUX bakes hard shadow contours into the geometry. Pass 2 then sharpens that baked edge into an unnatural vertical seam across the face.
* **The Solution:** Use `StudioPromptSanitizer` with the **Pass 1 Albedo Suffix** (`", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog"`). Pass 1 generates clean, unlit geometry, giving `StudioLightRig` 100% uncontaminated control over every photon.

### Law 2: Matte Fabrics Require `specular_kick = 0.00`
* **The Problem:** Dark wool coats have an albedo near $0.02$. High specular kicks ($> 2.0$) with low ambient floors cause grazing Fresnel spikes, making black wool look like metallic foil or thermal imaging.
* **The Solution:** Keep `specular_kick` modest ($0.30 – 0.50$) with the dedicated `specular_pass` wired. Direct diffuse radiance still highlights patent leather and metal hardware without solarizing matte wool fibers.

### Law 3: Rear Lights Require Forward Beam Coning
* **The Problem:** Omnidirectional mathematical point lights placed behind the subject illuminate background architecture behind the light stand at $160^\circ+$ off-axis, creating severe edge halos.
* **The Solution:** The optical forward modifier beam cone ($\cos\theta_{\text{beam}} > 0$) enforces real studio fixture behavior: light travels forward towards the subject, and zero light spills backward onto background arches.

### Law 4: Pass 2 Denoise Governs Light Integration
* **`0.35 – 0.45` (Identity Lock / Soft Commercial):** 100% facial feature lock. Ideal for catalog beauty lighting with soft modifiers (Beauty Dish / Octabox).
* **`0.55 – 0.65` (Chiaroscuro Sweet Spot):** FLUX generatively re-synthesizes subsurface scattering on skin, new eye catchlights, and leather reflections while preserving human likeness.
* **`0.75+` (Heavy Hallucination Warning):** At high denoise, dark proxy corners combined with prompt tokens like `"Stripbox"` cause FLUX to hallucinate physical studio softboxes hanging in the frame. Keep denoise $\le 0.65$.

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
* **`StudioLight_MethodB_Trajectory.json`**: Single-pass continuous latent trajectory splitting and delta-injection at flow step $\sigma_{10} = 0.863338$.
* **`StudioLight_Midstream_Master.json`**: Midstream KSampler hook architecture for fine-grained Euler step interception.
* **`StudioLight_Midstream_Sweep.json`**: Automated parameter sweep across lighting angles.

---

## 📄 License
MIT License. Created by [Maciej Tajber](https://github.com/maciejtajber).
