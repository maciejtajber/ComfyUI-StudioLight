# ComfyUI-StudioLight

> **Professional 3-Point Photographic Studio Lighting Console & Virtual Photon Engine for ComfyUI.**  
> Stop gambling with prompt RNG. Take absolute mathematical control over Key, Fill, and Rim lights in generative diffusion with 3D camera-space back-projection, real-world Kelvin temperatures, optical modifiers, inverse-square $(1/r^2)$ distance falloff, live 2D radar gizmos, and physical photometric compositing.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Optimized for: Flux.1 Dev | Flux.2 | SDXL](https://img.shields.io/badge/Optimized%20for-Flux.1%20%7C%20Flux.2%20%7C%20SDXL-black?style=flat-square)
![Tests: 14/14 Passed](https://img.shields.io/badge/Automated%20Suites-14%2F14%20Verified-brightgreen?style=flat-square)
![Status: Production Calibrated](https://img.shields.io/badge/Status-Production%20Calibrated-success?style=flat-square)

---

## The Breakthrough

> **"Photoshop paints 2D pixels. StudioLight simulates 3D virtual photons."**  

In standard generative AI, typing `"dramatic golden hour rim light"` is pure lottery—words bleed into facial anatomy and cannot guarantee where photons strike. 

**ComfyUI-StudioLight bridges true 3D studio photography into generative diffusion models:**
1. **3D Pinhole Camera Back-Projection ($40^\circ$ Portrait FOV):** Unprojects 2D surface normals and depth maps into true OpenGL metric coordinates ($Z = -1.5\text{m}$ to $-4.5\text{m}$) with anamorphic non-square aspect ratio correction.
2. **Photographic Power Falloff ($1/d^{1.6}$) & Hermite Distance Taper:** Computes per-light Euclidean stand distance falloff with a smooth background distance taper ($d_{\text{norm}} < 0.20$), preventing studio strobes from bleeding onto distant architectural facades.
3. **Optical Forward Modifier Beam Cone:** Gates light emission forward towards the subject via modifier beam coning, completely eliminating backward light leakage and architectural edge halos.
4. **Dedicated Specular Pass & Rational Shoulder Tonemapping:** Isolates a crisp Blinn-Phong dielectric catchlight lobe (2.8% pixel area) mapped through a $C^1$-continuous rational shoulder with active `white_point` headroom control.
5. **Automatic Albedo Prompt Decoupling (`StudioPromptSanitizer`):** Automatically strips high-specular tokens (`"glossy"`, `"patent"`, `"chrome"`) exclusively for Pass 1 (Steps 0–10) while preserving the complete master prompt for Pass 2, eliminating pre-baked specular blobs on patent leather and metallic surfaces.
6. **Two Battle-Tested Architectures:**
   * **Method A (2-Stage Calibrated Relighting):** The production workhorse. Combines unlit albedo draft passes with deferred 3D lit proxy plates and stochastic re-diffusion.
   * **Method B (Continuous Trajectory Injection):** Mathematical flow-matching split at step $\sigma_{10} = 0.863338$, intercepting clean $x_0$ via native Euler callback and injecting the lighting delta $\Delta x = (x_0' - x_0) \cdot \text{strength}$ directly into the ODE path.

---

## 📦 The 5-Node Core Suite

```text
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│     Studio3DStage      │ ───► │      StudioLightRig     │ ───► │StudioPhotometricComposite│
│ (3D Viewport / Presets)│      │ (3-Point Lighting Console)│     │(Linear Rec.709 Pipeline)│
└────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                              │
                                                                 ┌────────────▼────────────┐
                                                                 │   SIShadowHighlight     │
                                                                 │(Tonal Latent Sculptor)  │
                                                                 └─────────────────────────┘
                                                                              │
                                                                 ┌────────────▼────────────┐
                                                                 │   Pass 2 KSampler       │
                                                                 │ (Stochastic Polish)     │
                                                                 └─────────────────────────┘
```

### 1. `StudioLightRig` (`nodes/studio_light.py`)
Primary 3-point photographic console:
* **Interactive 2D Studio Radar:** Visual top-down radar on the node canvas with click-and-drag angle adjustment.
* **3-Point Control:** Independent Azimuth ($-180^\circ$ to $+180^\circ$), Elevation ($-30^\circ$ to $+90^\circ$), Distance ($0.5\text{m}$ to $10.0\text{m}$), and Intensity ($0.0$ to $3.0$) for Key, Fill, and Rim lights.
* **Photographic Modifiers:** Octabox (150cm), Silver Beauty Dish, Fresnel Spotlight, Hard Snoot, Stripbox, and Direct On-Camera Flash.
* **Blackbody Radiation Science:** Precision Kelvin curves ($2000\text{K} - 10000\text{K}$) mapped to IEC 61966-2-1 linear Rec.709 space.
* **Dual-Sphere VFX Preview:** Real-time 18% Grey Ball + Mirror Chrome Ball + Studio Radar card rendered in 0.03s.
* **Curated Presets:** `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.
* **High-Signal Prompt Token Synthesizer:** Auto-generates exact photographic direction tokens for FLUX, SDXL, and Video models (Wan/MiniMax).

### 2. `StudioPhotometricComposite` (`nodes/photometric_composite.py`)
Linear scene-referred photographic composite engine:
* **Zero Double-Floor Architecture:** Operates strictly in linear physical radiance space.
* **Rational Shoulder Tonemapping:** Smooth $C^1$-continuous shoulder curve that maps high dynamic range specular overshoots into headroom without hard-clipping.
* **Active `white_point` Control:** Dial dynamic range stiffness from 1.5 (tight ceiling) to 8.0 (extended highlights).
* **Dielectric Fresnel Catchlight Gate:** Guarantees specular catchlights survive on black patent leather and dark eyes ($F_0 \ge 0.12$).
* **Chroma Bleed Control:** Safely injects Kelvin color warmth without contaminating dark fabrics.

### 3. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
Latent-level digital darkroom tonal sculptor:
* **`shadow_amount`**: Deepens shadows into rich noir blacks (`-0.30`) or lifts dark fabrics (`+0.25`).
* **`highlight_amount`**: Boosts specular catchlights on jewelry, metallic chains, and glossy leather.
* **`shadow_tonal_width` / `highlight_tonal_width`**: Shapes the transition zone into midtones.
* **`midtone_contrast`**: Tunes punch without blowing out highlight detail.

### 4. `Studio3DStage` (`nodes/studio_3d_stage.py`)
Universal 3D asset loader and stage anchor:
* **Built-in Mannequin Assets:** Head, Bust, Medium, and Full-body mannequins.
* **Custom 3D Mesh Support:** Instant loading of `.glb` / `.gltf` assets for products, footwear, and automotive lighting.
* **Metadata Carrier:** Transmits scene metadata and coordinate convention flags directly to `StudioLightRig`.

### 5. `StudioLightTrajectory` Nodes (`nodes/trajectory_nodes.py`)
Method B mathematical diffusion flow injection:
* **`StudioLightTrajectorySplitter`**: Executes diffusion from step 0 to split-step (calibrated at $\sigma_{10} = 0.863338$), intercepting un-overshot clean prediction $x_0$ via native Euler callback.
* **`StudioLightTrajectoryInject`**: Combines relit latent $x_0'$ with noisy ODE state $x_t$ via $\text{injected} = x_t + (x_0' - x_0) \cdot \text{strength}$.

---

## 🔬 Empirical Discoveries & Forensic Rules

Through exhaustive empirical calibration on FLUX Krea Dev across 30+ iterations, we established 3 fundamental laws of generative lighting:

### Law 1: Pass 1 Must Be a Flat Albedo Carrier
* **The Problem:** If Pass 1 is prompted with directional sunlight or contrast, FLUX bakes hard shadow contours into the geometry. Pass 2 then sharpens that baked edge into an unnatural vertical seam across the face.
* **The Solution:** Use the **Pass 1 Albedo Lighting Suffix** (`", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog"`). Pass 1 generates clean, unlit geometry, giving `StudioLightRig` 100% uncontaminated control over every photon.

### Law 2: Matte Fabrics Require `specular_kick = 0.00`
* **The Problem:** Dark wool coats have an albedo near $0.02$. High specular kicks ($> 2.0$) with low ambient floors cause grazing Fresnel spikes, making black wool look like metallic foil or thermal imaging.
* **The Solution:** Set `specular_kick = 0.00` and `chroma_bleed = 0.00` when relighting dark matte clothing. Direct diffuse radiance still highlights patent leather and metal hardware without solarizing wool fibers.

### Law 3: Pass 2 Denoise Governs Light Integration
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
* **`StudioLight_v0.1_Starter.json`**: Lean, single-click starter workflow. Features the clean 3-plate display (Pass 1 Albedo Draft $\to$ 3D Lit Proxy Plate $\to$ Final Master Render) with diagnostic clutter stripped for seamless production use.
* **`StudioLight_MethodA_Calibrated.json`**: The calibrated production standard (37 nodes with visual engineering notes and organized zones). Features FLUX Krea Dev, DSINE 3D Normals, Depth Anything V2, dual-conditioning prompt synthesis, and live preview cards.
* **`StudioLight_MethodB_Trajectory.json`**: Single-pass continuous latent trajectory splitting and delta-injection at flow step $\sigma_{10} = 0.863338$.
* **`StudioLight_Midstream_Master.json`**: Midstream KSampler hook architecture for fine-grained Euler step interception.
* *(Legacy and pre-refactor exploratory graphs have been cleanly quarantined to `workflows/archive/`)*.

---

## 📄 License
MIT License. Open-source for personal, commercial, and research use. Designed and engineered for high-end fashion, cinematic portraits, and luxury synthetic identity.
