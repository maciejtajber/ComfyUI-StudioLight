# ComfyUI-StudioLight

> **Professional Photographic 3-Point Studio Lighting Console & Photometric Engine for ComfyUI.**  
> Direct the physical illumination of generative diffusion models with spatial control. Position Key, Fill, and Rim lights in 3D camera-space, dial blackbody-inspired Kelvin temperatures, shape beam angles with optical modifiers, control photographic distance falloff, inspect live 2D radar gizmos, and composite with scene-referred linear photometric math.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Calibrated for: Flux.1 Dev | SDXL](https://img.shields.io/badge/Calibrated%20for-Flux.1%20Dev%20%7C%20SDXL-black?style=flat-square)
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
The linear physics engine translating 3D spatial vectors into physical photons.

This node is not a simple image overlay or Photoshop blend mode. It performs scene-referred linear Rec.709 compositing, taking raw geometric vectors (normals, depth, light distance, and modifier profiles) and computing real physical radiance before display tonemapping:

* **Zero Double-Floor Architecture:**  
  Unlike typical post-processing nodes that apply arbitrary ambient lifts, it calculates diffuse illumination in true linear space. Surfaces maintain genuine contrast ratios without grey, washed-out shadow floors.
* **Rational Shoulder Tonemapping:**  
  High-intensity studio strobes produce high dynamic range specular spikes. A $C^1$-continuous rational shoulder tonemapper compresses specular peaks into display headroom smoothly, preserving texture without hard digital clipping or solarized white halos.
* **Lobe-Accurate Specular Integration (`specular_pass` $\to$ `specular_map`):**  
  Accepts the isolated Blinn-Phong specular pass from `StudioLightRig`. This confines dielectric catchlights exclusively to genuine specular microfacets (eyes, lips, patent leather, polished metal), preventing specular light from washing out matte fabric textures.
* **Blackbody Kelvin Midtone Bleed:**  
  Controls how color temperature transfers into surface albedo via `chroma_bleed`. Warms or cools skin tones naturally according to dialed Kelvin values ($2000\text{K} - 10000\text{K}$) without contaminating neutral shadows or dark garments.

#### Core Shader Controls:
* **`lighting_strength` (`1.00 – 1.20`):** Master irradiance multiplier. `1.00` provides physically exact photometric scale; `1.20` gives a slight photographic push for high-key portraiture.
* **`ambient_floor` (`0.05 – 0.08`):** Minimum shadow illuminance floor. Prevents unlit surface cavities from collapsing to pitch black ($0.0$), ensuring the diffusion model has underlying structure to build upon.
* **`specular_kick` (`0.40 – 0.50`):** Dielectric catchlight amplitude. Calibrated to `0.40` for mixed scenes (skin + leather + wool). Set to `0.00` for pure matte black wool to eliminate edge solarization.
* **`chroma_bleed` (`0.35 – 0.45`):** Kelvin warmth midtone transfer. Default `0.45` delivers balanced portrait saturation. Keep at `0.00` for monochrome or high-contrast noir setups.
* **`filmic_white_point` (`2.5 – 3.5`):** Dynamic range compression ceiling. `2.5` creates a soft, creamy highlight roll-off (catalog beauty); `3.5` maintains crisp specular punch on jewelry, chrome, and patent leather.

---

### 3. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
The digital darkroom lathe adapting physical plates for latent diffusion.

Diffusion UNETs and Transformers do not perceive light like physical camera sensors. When fed raw linear render plates, diffusion models often misinterpret pure zero-radiance shadows as missing data or sensor noise, leading to smearing, muddy gray voids, or hallucinated background clutter in Pass 2. 

`SIShadowHighlight` functions as an adaptive dodging-and-burning stage directly preceding `VAE Encode Lit Plate`, shaping the photometric plate into optimal tonal ranges for latent conditioning:

* **Adaptive Shadow Conditioning (`shadow_amount`):**  
  * *Negative values (`-0.15` to `-0.25`):* Pulls ambient shadow zones into dense, inky noir blacks without destroying edge geometry.
  * *Positive values (`+0.10` to `+0.20`):* Gently recovers deep shadow crevices (e.g. under dark lapels or folded sleeves), feeding sufficient surface density to FLUX so Pass 2 synthesizes authentic fabric weave rather than muddy noise.
* **Specular Beaconing (`highlight_amount: 0.15`):**  
  Applies a targeted micro-boost strictly across high-energy specular microfacets. This acts as a high-frequency attention beacon for the Pass 2 diffusion process, directing FLUX to synthesize sharp catchlights in the pupils, wet lips, and metallic buckles.
* **Bilateral Tonal Width Isolation:**  
  `shadow_tonal_width` and `highlight_tonal_width` (calibrated at `0.50`) lock adjustments to the extreme ends of the luminance histogram, leaving skin midtones and facial features untouched.
* **Perceptual Midtone S-Curve (`midtone_contrast`):**  
  Tunes perceptual punch and contrast slope without shifting the white point or shadow floor.

### 4. `StudioPromptSanitizer` (`nodes/prompt_sanitizer.py`)
Automatic albedo decoupling engine:
* Strips high-gloss/specular terms from Pass 1 while delivering the full user prompt to Pass 2.
* Mitigates pre-baked specular blobs on patent leather, glass, and wet surfaces before they bake into the latent carrier.

---

## 💡 Photographic Modifiers & Optical Physics

In physical studio lighting, light character is governed by a fundamental optical principle: **the physical diameter of the light source relative to the subject dictates shadow edge penumbra, specular highlight shape, and light wrap.**

`StudioLightRig` does not treat light sources as dimensionless mathematical points. Each modifier dropdown assigns a real-world physical aperture diameter to the virtual strobe, driving our **Screen-Space Percentage-Closer Soft Shadow (PCSS)** engine and Blinn-Phong dielectric specular lobes.

---

### The 6 Optical Presets

| Modifier Preset | Virtual Aperture | Optical Profile | Shadow Behavior | Primary Aesthetic |
| :--- | :---: | :--- | :--- | :--- |
| **Direct On-Camera Flash** | **0.04m** (~4cm) | Raw, bare bulb / speedlight tube | Razor-sharp edges across all distances | High-contrast, raw 90s editorial / Terry Richardson |
| **Hard Snoot** | **0.10m** (~10cm) | Narrow conical metal restriction tube | Hard boundary cutoff, zero lateral spill | Theatrical spotlight, isolated facial/jewelry drama |
| **Fresnel Spotlight** | **0.20m** (~20cm) | Collimated stepped glass lens | Sharp, sculpted edges with subtle edge roll | 1940s Hollywood Golden Age portraits, Film Noir |
| **Stripbox** | **0.30m** (~30×120cm)| Asymmetric baffled softbox with grid | Directional wrap, controlled spill | Clean rim/kicker separation, edge silhouette accent |
| **Beauty Dish (Silver)** | **0.55m** (~55cm) | Parabolic reflector with center deflector plate | Defined bone-structure carving, soft skin falloff | High-fashion cosmetic advertising, magazine covers |
| **Octabox (Soft)** | **1.40m** (~140cm) | 5-foot 8-sided diffused umbrella | Broad, creamy, continuous penumbra gradient | Flattering catalog commercial, soft ambient window light |

---

### Detailed Optical Breakdown

#### 1. Direct On-Camera Flash (Aperture: 0.04m)
* **Studio Configuration:** A naked speedlight or bare strobe tube aimed straight along the camera axis without diffusion baffles.
* **Optical Behavior:** Near-point-source radiation.
* **Cast Shadows:** Razor-sharp boundary cutoffs across all scene depths with zero edge feathering.
* **Specular Catchlights:** Tiny, concentrated, high-intensity pinpoint catchlights in eyes and reflective hardware.
* **Aesthetic:** Unfiltered, direct fashion editorial.

#### 2. Hard Snoot (Aperture: 0.10m)
* **Studio Configuration:** An elongated cylindrical/conical metal snoot mounted directly to the fixture.
* **Optical Behavior:** Restricts beam divergence into a narrow circular cone with steep radial attenuation.
* **Cast Shadows:** Sharp geometric projection lines; un-aimed geometry drops off rapidly outside the cone.
* **Spill:** Zero lateral ambient bleed into surrounding architecture.
* **Aesthetic:** High-drama chiaroscuro, isolating specific planes (e.g. cheekbone, handbag buckle) while sinking the environment into darkness.

#### 3. Fresnel Spotlight (Aperture: 0.20m)
* **Studio Configuration:** Traditional studio tungsten housing paired with a stepped Fresnel glass lens.
* **Optical Behavior:** Parallel collimated beam emitting punchy, directional illumination.
* **Cast Shadows:** Clean, defined cast shadows that maintain edge clarity without the clinical harshness of a bare bulb.
* **Aesthetic:** Classical cinematic lighting, hard-boiled film noir, and sculpted architectural portraits.

#### 4. Stripbox (Aperture: 0.30m)
* **Studio Configuration:** A narrow, tall rectangular softbox fitted with front diffusion fabric and a directional honeycomb grid.
* **Optical Behavior:** Directional diffusion—softer along the vertical orientation, razor-sharp along the horizontal axis.
* **Cast Shadows:** Controlled directional cutoff.
* **Specular Catchlights:** Long, continuous linear highlights tracing curved surfaces (e.g. cylinder folds of a wool coat or patent leather flaps).
* **Aesthetic:** Rim and kicker isolation designed to separate dark garments and silhouettes from dark backgrounds without leaking onto facial planes.

#### 5. Beauty Dish - Silver (Aperture: 0.55m)
* **Studio Configuration:** Shallow parabolic metal reflector with an opaque center bounce plate that forces all light to bounce off the silver dish interior before exiting.
* **Optical Behavior:** Semi-hard, directional contrast with accelerated highlight-to-shadow falloff.
* **Cast Shadows:** Crisp enough to carve out jawlines and cheekbone hollows, but soft enough to flatter human skin.
* **Specular Catchlights:** Distinct, punchy circular catchlights in the irises.
* **Aesthetic:** Standard commercial high-fashion, cosmetics campaigns, and luxury editorial portraiture.

#### 6. Octabox - Soft (Aperture: 1.40m)
* **Studio Configuration:** A 140cm octagonal softbox incorporating both internal baffle fabric and a heavy outer diffusion panel.
* **Optical Behavior:** Large virtual surface area emitting widely diffused light rays that wrap around curved geometry.
* **Cast Shadows:** Broad, creamy penumbra. Cast shadows dissolve over wide pixel gradients as the distance from subject to background increases.
* **Specular Catchlights:** Large, smooth, soft reflection panels across glossy materials and skin.
* **Aesthetic:** Natural beauty catalog, gentle overcast simulation, and soft editorial fill.

---

### What Changes in the Shading Engine When Modifiers Switch

Toggling a modifier updates three physical calculations simultaneously:

1. **Percentage-Closer Soft Shadows (PCSS):**  
   The engine calculates the spatial gap between occluder geometry and receiver planes ($\Delta z = d_{\text{receiver}} - d_{\text{blocker}}$) and scales an 8-tap Vogel disk sampling kernel by modifier diameter. Contact shadows (e.g., collar on neck, chin on throat) remain crisp and contact-hardened, while shadows cast onto background architecture $2\text{m}$ to $5\text{m}$ back softly diffuse according to aperture width.
2. **Specular Reflection Lobe Width:**  
   Dielectric catchlights expand from a tiny pinpoint (Direct Flash / Snoot) into a broad, soft reflective window (Octabox), matching the virtual surface area of the fixture.
3. **Forward Optical Beam Coning:**  
   The optical housing limits backward light spill ($\cos\theta_{\text{beam}} > 0$). Fixtures aimed forward at a model will not leak backward onto background walls or arches located behind the virtual light stand.

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

## 🎛️ Recommended Settings & Production Recipes

Use these empirically calibrated starting points inside `StudioPhotometricComposite` and `StudioLightRig` to achieve immediate photographic results:

### 1. Production Recipes by Scenario

| Setup / Scenario | `specular_kick` | `chroma_bleed` | `filmic_white_point` | Pass 2 Denoise | Recommended Key Modifier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **High-Fashion Beauty (Skin & Soft Fill)** | `0.40 – 0.50` | `0.40 – 0.45` | `3.5` | `0.50 – 0.55` | `Beauty Dish (Silver)` or `Octabox` |
| **Commercial Catalog (Mixed Fabrics & Leather)** | `0.40 – 0.50` | `0.35` | `3.5` | `0.55 – 0.60` | `Beauty Dish` or `Stripbox` |
| **Dark Tailored Apparel (Matte Wool / Velvet)** | `0.00 – 0.20` | `0.00` | `2.5` | `0.55 – 0.60` | `Octabox (Soft)` |
| **Cinematic Film Noir (High Chiaroscuro)** | `0.50 – 0.60` | `0.50` | `2.0 – 2.5` | `0.60 – 0.65` | `Fresnel Spotlight` or `Hard Snoot` |
| **Isolated Metallic / Chrome Props (Macro)** | `1.00 – 1.50` | `0.10` | `5.0 – 6.0` | `0.60 – 0.65` | `Direct Flash` or `Hard Snoot` |

> **⚠️ Multi-Material Rule:** When dark matte wool and glossy patent leather appear in the same shot, keep `specular_kick` at **`0.40 – 0.50`**. With `specular_pass` wired to `specular_map`, the isolated Blinn-Phong lobe concentrates crisp catchlights onto leather and eyes without solarizing dark wool fibers.

---

### 2. The 4 Core Composite Sliders Decoded

* **`specular_kick` (`0.0 – 3.0`):**  
  Controls the amplitude of dielectric specular catchlights.
  - *Keep at `0.00` for pure matte garments* to guarantee zero edge solarization.
  - *`0.40 – 0.50` (Calibrated Standard):* The universal sweet spot for portraits containing mixed materials (skin, eyes, wool, leather).
  - *`1.0+`:* Reserve strictly for isolated metallic/jewelry product shots.  
  *(Always ensure `specular_pass` from `StudioLightRig` is connected to `specular_map`!)*

* **`chroma_bleed` (`0.0 – 1.0`):**  
  Controls how strongly Kelvin color temperature (e.g. 3200K Tungsten or 7000K Cyan) transfers into surface midtones.
  - *`0.40 – 0.45` (Default):* Natural photographic color balance.
  - *Set to `0.00`:* When you want pure monochrome exposure without tinting dark clothing or neutral shadows.

* **`filmic_white_point` (`1.5 – 8.0`):**  
  Governs highlight dynamic range compression in linear Rec.709 space.
  - *`2.0 – 2.5`:* Tight photographic ceiling. Soft, creamy highlight roll-off (ideal for delicate skin and overcast beauty).
  - *`3.5` (Default):* Balanced filmic curve with natural highlight preservation.
  - *`5.0 – 8.0`:* High dynamic headroom. Preserves sharp specular glints on polished chrome and direct flash reflections.

* **`ambient_floor` (`0.0 – 0.50`):**  
  Sets the baseline shadow illuminance floor.
  - *Keep at `0.00`:* For true optical contrast, deep chiaroscuro, and rich blacks.
  - *Lift to `0.05 – 0.08`:* Only if Pass 2 over-darkens tight cavity crevices on high-contrast prompts.

---

### 3. Quick-Start: Your First Relit Portrait in 60 Seconds

1. Load `workflows/StudioLight_MethodA_Calibrated.json` into ComfyUI.
2. In **Zone 1**, enter your subject description in the **Base Subject Prompt** (leave the Albedo Lighting Suffix connected).
3. On **`StudioLightRig`**, choose a starting preset (e.g., `High-Fashion Beauty` or `Rembrandt Classic`), or position Key/Fill/Rim manually.
4. Pass 1 is pre-set to **10 steps** (draft albedo geometry); Pass 2 is pre-set to **30 steps** with denoise at **`0.55 – 0.60`**.
5. Click **Queue Prompt**. Inspect the live side-by-side preview cards: `3D Surface Normals`, `Depth Map`, `3D Lit Proxy Plate`, and the `Final Master Render`.

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
