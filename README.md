<div align="center">

<img src="assets/studiolight_banner.png" alt="StudioLight by METAMATERIAL" width="100%" />

<br>
<br>

**Stop prompt gambling. Design the light.**

Camera-space key, fill and rim lights, physical modifiers and Kelvin control for FLUX in ComfyUI.

<sub>V1.0.0 &nbsp;·&nbsp; COMFYUI CUSTOM NODE &nbsp;·&nbsp; FLUX.1 DEV / KREA &nbsp;·&nbsp; 18/18 TESTS &nbsp;·&nbsp; MIT</sub>

<br>

<sub>
<a href="#the-ground-truth-deterministic-3d-guidance-meets-latent-diffusion">THE PROOF</a> &nbsp;·&nbsp;
<a href="#what-makes-studiolight-different">DIFFERENTIATION</a> &nbsp;·&nbsp;
<a href="#same-subject-same-prompt-different-light">SHOWCASE</a> &nbsp;·&nbsp;
<a href="#how-it-works">PIPELINE</a> &nbsp;·&nbsp;
<a href="#metric-3d-camera-space-shadow-raymarching">3D SHADOWS</a> &nbsp;·&nbsp;
<a href="#optical-modifiers--physical-area-emitters">AREA EMITTERS</a> &nbsp;·&nbsp;
<a href="#the-core-production-suite-deep-technical-manual">MANUAL</a> &nbsp;·&nbsp;
<a href="#installation">INSTALL</a> &nbsp;·&nbsp;
<a href="#master-workflows">WORKFLOWS</a>
</sub>

</div>

<br>

---

## The Ground Truth: Deterministic 3D Guidance Meets Latent Diffusion

> **Identical Camera Intrinsics ($40^\circ$ Portrait FOV) & Studio Rig Coordinates.**  
> Left: Raw 32-bit metric deferred lighting pass. Right: Final FLUX.1 Krea Dev latent synthesis.

<table>
  <thead>
    <tr>
      <th width="50%" align="center">1. Metric Deferred Shading Pass (PyTorch Raymarcher)</th>
      <th width="50%" align="center">2. Final FLUX Synthesis (Physical Denoising)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img src="assets/hero_split_deferred.png" width="100%" alt="Deferred 3D Shading Pass" /></td>
      <td align="center"><img src="assets/hero_split_flux.jpg" width="100%" alt="Final FLUX Latent Synthesis" /></td>
    </tr>
    <tr>
      <td align="left">
        <b>Deterministic Physical Guidance:</b><br>
        • Canonical 3D camera-space raymarching through metric depth field<br>
        • Optical contact-hardening under knuckles and coat collar<br>
        • Diverging physical penumbra cast across 2m background chasm
      </td>
      <td align="left">
        <b>Photorealistic Material Interpretation:</b><br>
        • Patent leather dielectric specular catchlights (zero grey veil)<br>
        • Deep luxury black wool texture (zero clipping or self-shadow acne)<br>
        • Fluted neoclassical stone grain and ambient occlusion
      </td>
    </tr>
  </tbody>
</table>

---

## What Makes StudioLight Different

ComfyUI-StudioLight integrates image-derived camera-space geometry, metric screen-space shadow raymarching, and sampled photographic area emitters into a controllable lighting workflow for generative image synthesis.

### Defining Capabilities:
* **Metric Camera-Space Shadowing:** Reconstructs canonical 3D surface positions from image-derived depth and camera intrinsics ($40^\circ$ FOV), evaluating visibility along projected 3D light rays in meters rather than flat screen-space offsets.
* **Geometric Area-Emitter Sampling:** Models physical studio modifier apertures (Octabox 1.4m, Stripbox 0.3×1.2m, Beauty Dish 55cm) through Quasi-Monte Carlo Hammersley sampling. Contact-hardening penumbras emerge naturally from sample integration without artificial PCSS blur passes.
* **Integrated Photometric Lighting:** Combines directional diffuse shading, Blinn-Phong specular microfacets, distance-dependent $(r_{\text{ref}}/r)^2$ radiometric falloff, and shadow visibility within a unified PyTorch pipeline.
* **Diffusion-Integrated Results:** Uses the computed lighting plate as explicit physical guidance for Pass 2 FLUX.1 Krea Dev latent synthesis, marrying spatial lighting control with generative material synthesis.

> [!NOTE]
> **Scope & Engineering Limits:** StudioLight performs metric screen-space raymarching over an estimated single-view geometry field (Depth Anything V2 + DSINE). It is designed for fast, controllable image relighting—not unrestricted path tracing over a complete multi-view 3D scene.

### Context & Adjacent Prior Art

StudioLight builds on established computer-graphics principles and exists alongside valuable ComfyUI lighting tools:
* **Learned Relighting (e.g., [IC-Light](https://github.com/lllyasviel/ic-light)):** Employs text- and background-conditioned diffusion models. StudioLight complements learned approaches by offering explicit physical studio fixture parameters (stand distance in meters, elevation/azimuth in degrees, blackbody Kelvin, physical modifier geometry).
* **Scene-Linear Point Lighting (e.g., [FXTD Radiance](https://github.com/FXTD-Studios/radiance)):** Features excellent scene-linear math and camera-space point lights, accepting external shadow passes. StudioLight provides an integrated geometric area-emitter shadow raymarcher and single-view metric reconstruction.
* **Full 3D Environments (e.g., [WAS Node Suite Three.js](https://github.com/WASasquatch/was-node-suite-comfyui)):** Provides full 3D polygonal mesh path tracing. StudioLight focuses specifically on single-view generative images where no 3D polygonal mesh exists.
* **Screen-Space Mask Lighting (e.g., [ComfyUI-ReLight](https://github.com/EnragedAntelope/comfyui-relight)):** Performs fast 2D mask-aware raymarching. StudioLight extends this from 2D mask space into metric 3D camera space with depth-buffer occlusion testing.

---

## Same Subject. Same Prompt. Different Light.

> **Locked Seed:** `424242` | *Identical Base Prompt, Geometry, and Seed across FLUX.1 Krea Dev*

<table>
  <thead>
    <tr>
      <th width="33.33%" align="center">Hard Key (Fresnel)</th>
      <th width="33.33%" align="center">Sculpted (Beauty Dish)</th>
      <th width="33.33%" align="center">Soft Wrap (Octabox)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img src="assets/showcase_fresnel.png" width="100%" alt="Fresnel Spotlight" /></td>
      <td align="center"><img src="assets/showcase_beauty.png" width="100%" alt="Beauty Dish" /></td>
      <td align="center"><img src="assets/showcase_octabox.png" width="100%" alt="Diffused Octabox" /></td>
    </tr>
    <tr>
      <td align="center">
        <b>Fresnel Spotlight (Hard Key)</b><br>
        Key: +30° Azimuth / +25° Elevation<br>
        Crisp penumbra, sharp dielectric glints
      </td>
      <td align="center">
        <b>Beauty Dish (Silver 55cm)</b><br>
        Key: +30° Azimuth / +25° Elevation<br>
        Carved cheekbones, crisp eye catchlights
      </td>
      <td align="center">
        <b>Diffused Octabox (140cm)</b><br>
        Key: +30° Azimuth / +25° Elevation<br>
        Soft continuous penumbra, creamy coat wrap
      </td>
    </tr>
  </tbody>
</table>

---

## The Problem: Prompt Lottery vs. Spatial Control

Text prompts cannot reliably execute 3D spatial vectors. Requesting a `"dramatic blue rim light on the left"` often triggers severe prompt bleed: facial features shift, clothing changes cut, and colors wash across the entire frame.

| Prompt-Based Lighting (Text Only) | StudioLight Spatial Control (VFX Engine) |
| :--- | :--- |
| **Subject Mutation:** Prompting for light changes anatomy, styling, and background geometry. | **Geometry Preservation:** Structure crystallizes in Pass 1; lighting is steered independently in Pass 2. |
| **Chromatic Bleed:** Color tokens tint skin, hair, and clothing indiscriminately. | **Directional Lobe Isolation:** Rim and key colors remain confined to illuminated surfaces ($N \cdot L > 0$). |
| **Uncalibrated Falloff:** Lights fail to observe inverse-square distance or scene depth. | **Physical Geometric Inverse-Square:** Models true $(r_{\text{ref}}/r)^2$ radiometric distance falloff, normalized to studio reference stand distance ($r_{\text{ref}}=1.8\text{m}$). |
| **Pre-baked Specular Artifacts:** High-gloss words bake blown-out white patches into early draft latents. | **Albedo Decoupling:** Renders clean, shadowless draft plates; specular highlights ignite only where fixtures strike. |

---

## How It Works

StudioLight splits generation into a **two-stage geometry and relighting pipeline**:

<img src="assets/studiolight_pipeline.png" alt="StudioLight pipeline: prompt, draft, geometry, light rig, composite, darkroom, refine" width="100%" />

1. **Geometry Crystallization (Steps 0–10):** Pass 1 generates an unlit albedo draft to define scene forms, anatomy, and clothing weave without baked directional shadows.
2. **Metric 3D Projection:** Extracts surface normals (DSINE) and metric depth (Depth Anything V2), back-projecting the 2D draft into virtual camera-space ($40^\circ$ portrait FOV).
3. **Photometric Deferred Shading:** Computes diffuse irradiance, Blinn-Phong specular lobes (tight 2.8% pixel area), and metric 3D camera-space shadow raymarching from virtual C-stands.
4. **Stochastic Polish (Steps 10–30):** Composites the lit plate in linear Rec.709 space, preps tones in the `SIShadowHighlight` darkroom, and feeds it to Pass 2 (denoise: `0.55 – 0.60`) to synthesize subsurface skin scattering and photorealistic catchlights.

---

## Metric 3D Camera-Space Shadow Raymarching

StudioLight replaces flat screen-space offsets with **canonical 3D camera-space raymarching through reconstructed metric depth fields**:

```
P_receiver ──► P_receiver + t · (P_light - P_receiver) ──► Perspective Projection ──► Bilinear Depth Lookup ──► P0 Occlusion Gate
```

* **P0 Front-of-Ray Occlusion Gate ($D_{\text{scene}} \le D_{\text{ray}} - \epsilon$):** Evaluates depth strictly in front of the ray sample, eliminating false shadows and receiver self-shadow acne on curved faces while handling camera-side key lights and rear rim kickers symmetrically.
* **Unbounded 3D Depth Chasms:** Foreground subjects cast geometrically accurate shadows across 20m+ depth gaps onto distant architecture, deforming authentically across fluted columns and background masonry.
* **Single-Layer Physical Penumbra:** Soft shadows emerge naturally from multi-sample Quasi-Monte Carlo integration across the physical aperture face (Octabox 1.40m, Stripbox 0.30×1.20m, Beauty Dish 55cm) with **Interleaved Gradient Noise (IGN)** sub-step dither to eradicate banding—zero artificial PCSS post-blur required.

<div align="center">
<img src="assets/stress_02_slit_banding_vs_dither.png" width="95%" alt="Metric Shadow Raymarching: Banding vs IGN Dither" />
<br>
<sub>Slit Aperture Penumbra Stress Test (Row 1: S=4 no dither, S=8 no dither, S=4 dither. Row 2: S=8 dither [Balanced Default], S=16 dither [High-End], S=64 Quasi-Monte Carlo Ground Truth).</sub>
</div>

### NVIDIA RTX 3090 Performance Benchmarks

*Measured via `tools/run_hardcore_stress_tests.py` on an NVIDIA GeForce RTX 3090 (PyTorch 2.13, CUDA 12.6, 5 warm runs averaged with `torch.cuda.synchronize()`):*

| Resolution | Preset | Samples ($S$) | Steps | Latency (RTX 3090) | Peak VRAM | Mega Ray-Sample Steps / sec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$512 \times 512$** | Fast Preview | 4 | 12 | **12.0 ms** | 113 MB | $1,047.1 \times 10^6$ |
| | Balanced (Default) | 8 | 16 | **29.6 ms** | 118 MB | $1,135.1 \times 10^6$ |
| **$1024 \times 1024$** | Fast Preview | 4 | 12 | **32.0 ms** | 450 MB | $1,571.3 \times 10^6$ |
| | **Balanced (Default)** | **8** | **16** | **83.0 ms** | **469 MB** | **$1,616.9 \times 10^6$** |
| | High-End Quality | 16 | 24 | **245.3 ms** | 469 MB | $1,641.5 \times 10^6$ |
| **$2048 \times 2048$** *(4.2 MP)* | Balanced (Default) | 8 | 16 | **311.5 ms** | 1.87 GB | $1,723.5 \times 10^6$ |

> [!NOTE]
> For the complete mathematical formulation, low-discrepancy sampling proofs, and shader implementation:
> * [Mathematical Proof: Metric 3D Camera-Space Shadow Raymarching](docs/ARCHITECTURE_METRIC_3D_SHADOWS.md)
> * [Architecture Plan: StudioLight Geometric Area Emitters](docs/ARCHITECTURE_PHYSICAL_AREA_EMITTERS.md)
> * [Engine Walkthrough & Benchmarks](docs/ENGINE_WALKTHROUGH_AREA_EMITTERS.md)

---

## Optical Modifiers & Physical Area Emitters

StudioLight implements a **physically based geometric area-emitter architecture**. Every studio modifier is modeled as a defined 3D virtual emitter with metric dimensions ($W, H, R$), angular beam distributions, single-layer penumbra visibility, and an analytical aperture reflection in reflection space:

```
                                   EMITTER
        [Geometry: Dimensions in meters (W, H, R), Position, Aim Direction]
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   1. DIFFUSE EMISSION       2. SHADOW / VISIBILITY     3. SPECULAR REFLECTION
  Vectorized [S, H, W, 3]    Single-layer penumbra      Ray-plane intersection
  • S=16 sample points       • Binary blocker test      • Emitter 2D coordinate
  • Physical 1 / r_k²          per sample point         • Continuous aperture
  • cos θ_e × cos θ_p        • Average across S           profile & roughness
  • Radiometric area norm      yields natural penumbra    • True fixture catchlights
```

| Modifier | Metric Aperture | Beam Angle ($\theta_{\text{beam}}$) | Optical Model | Catchlight Profile |
| :--- | :---: | :---: | :--- | :--- |
| **Direct On-Camera Flash** | $0.08\text{m} \times 0.05\text{m}$ | $90^\circ$ | Near-point strobe emitter | Crisp micro-pinpoint |
| **Hard Snoot** | $R=0.05\text{m}$ (10cm) | $18^\circ$ | Steeper collimated beam cone | Tight circular isolation |
| **Fresnel Spotlight** | $R=0.15\text{m}$ (30cm) | $35^\circ$ | Optical lens edge cutoff | Compact stepped lens disc |
| **Stripbox** | $0.30\text{m} \times 1.20\text{m}$ | $65^\circ$ | $1:4$ Asymmetric rectangular emitter | Elongated vertical catchlight bar |
| **Beauty Dish (Silver)** | $R=0.275\text{m}$ (55cm) | $105^\circ$ | Parabolic dish + center deflector ($36\%$ ratio) | Sculpted cheekbones, donut/core ring |
| **Octabox (Soft)** | $R=0.70\text{m}$ (140cm) | $130^\circ$ | Broad Lambertian octagonal emitter | Large wrap disc, soft continuous penumbra |

---

## Showcase Gallery

Calibrated production renders executed on FLUX Krea Dev using Method A:

| Raw 90s Direct Flash (5500K) | Cinematic Fresnel Spotlight (5500K) |
| :---: | :---: |
| <img src="assets/showcase_flash.png" width="100%" alt="Direct Flash" /> | <img src="assets/showcase_fresnel.png" width="100%" alt="Fresnel Spotlight" /> |
| **Direct On-Camera Flash**<br>Key: 0° Azimuth / +5° Elevation (5500K)<br>Sharp contact drop shadow, punchy patent leather spec | **Fresnel Chiaroscuro**<br>Key: +30° Azimuth / +25° Elevation (5500K)<br>Collimated stepped beam, natural stone falloff |

| High-Fashion Beauty Dish (5500K) | Diffused Studio Octabox (5500K) |
| :---: | :---: |
| <img src="assets/showcase_beauty.png" width="100%" alt="Beauty Dish" /> | <img src="assets/showcase_octabox.png" width="100%" alt="Octabox" /> |
| **Beauty Dish (Silver 55cm)**<br>Key: +30° Azimuth / +25° Elevation<br>Carved cheekbone hollows, crisp eye catchlights | **Diffused Octabox (140cm)**<br>Key: +30° Azimuth / +25° Elevation<br>Soft, continuous penumbra, creamy coat wrap |

---

## The Core Production Suite (Deep Technical Manual)

### 1. `StudioLightRig` (`nodes/studio_light.py`)
The master 3-point photographic lighting console:
* **Interactive Lighting Console**: Visual top-down camera-space controls on the node canvas with click-and-drag angle adjustment.
* **3-Point Studio Control**: Independent Azimuth ($-180^\circ$ to $+180^\circ$), Elevation ($-30^\circ$ to $+90^\circ$), Stand Distance ($0.5\text{m}$ to $10.0\text{m}$), and Intensity ($0.0$ to $3.0$) for Key, Fill, and Rim lights.
* **Photographic Modifiers**: Octabox (150cm soft wrap), Silver Beauty Dish, Fresnel Spotlight, Hard Snoot, Stripbox, and Direct On-Camera Flash.
* **Blackbody-Inspired Kelvin**: Blackbody-inspired Kelvin-to-RGB conversion with linear-light Rec.709 compositing ($2000\text{K} - 10000\text{K}$).
* **Dual-Sphere VFX Preview**: Real-time 18% Grey Ball + Mirror Chrome Ball preview card rendered in 0.03s.
* **Curated Studio Presets**: `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.
* **4 Dedicated Outputs**: Exposes `relighting_map` (32-bit deferred plate), `specular_pass` (isolated Blinn-Phong lobes), `probe_preview`, and synchronized `lighting_prompt`.
* **Coordinate Parity Toggles**: `invert_normals_x` (DSINE / OpenGL parity), `invert_normals_y`, and `invert_depth` (near=white vs near=black polarity).

### 2. `StudioPhotometricComposite` (`nodes/photometric_composite.py`)
The linear physics engine translating 3D spatial vectors into physical photons:
* **Zero Double-Floor Architecture:** Calculates diffuse illumination in true linear space without washed-out grey shadow floors.
* **Rational Shoulder Tonemapping:** Compresses specular spikes into display headroom smoothly, preserving texture without hard digital clipping or solarized white halos.
* **Lobe-Accurate Specular Integration (`specular_pass` $\to$ `specular_map`):** Confines dielectric catchlights exclusively to genuine specular microfacets (eyes, lips, patent leather), leaving matte wool untouched.
* **Blackbody Kelvin Midtone Bleed:** Controls how color temperature transfers into surface albedo via `chroma_bleed` without polluting deep shadows.

### 3. `SIShadowHighlight` (`nodes/si_shadow_highlight.py`)
The digital darkroom lathe adapting physical plates for latent diffusion:
* **Adaptive Shadow Conditioning (`shadow_amount`):** Negative values (`-0.15` to `-0.25`) pull shadows into dense inky blacks; positive values gently recover deep crevices under dark lapels.
* **Specular Beaconing (`highlight_amount: 0.15`):** Applies a targeted micro-boost strictly across specular microfacets, directing FLUX attention to synthesize sharp catchlights in the pupils.
* **Bilateral Tonal Width Isolation:** Calibrated at `0.50` to lock adjustments strictly to extreme luminance ends, leaving skin midtones untouched.
* **Perceptual Midtone S-Curve (`midtone_contrast`):** Tunes perceptual contrast slope without shifting the white point.

### 4. `StudioPromptSanitizer` (`nodes/prompt_sanitizer.py`)
Automatic albedo decoupling engine: strips high-gloss and specular words from Pass 1 while delivering the full prompt to Pass 2, preventing pre-baked white solarization blobs in early latents.

---

## Empirical Discoveries & Working Rules

* **Rule 1: Pass 1 Functions Best as a Flat Albedo Carrier:** Prompting Pass 1 with directional sun bakes hard contours into base geometry. Use an albedo carrier suffix in Pass 1 (`", flat diffuse ambient lighting, shadowless, albedo"`) to give StudioLight clean geometric control.
* **Rule 2: Matte Fabrics Require Dedicated Specular Isolation:** Dark wool has albedo near 0.02. High specular kicks cause grazing Fresnel spikes, turning black wool into metallic foil. Wire `specular_pass` directly into `StudioPhotometricComposite`, keep `specular_kick` at `0.30–0.50`, and set `chroma_bleed = 0.00` on dark garments.
* **Rule 3: Radiometric Distance Falloff Replaces Artificial Depth Tapers:** Legacy screen-space shaders often clamp distant geometry (`d_norm < 0.20`) to suppress noise, which accidentally destroys shadows on distant walls. StudioLight evaluates unconstrained physical $(r_{\text{ref}}/r)^2$ radiometric falloff and directional beam profiles ($\cos\theta > 0$), allowing foreground subjects to naturally cast geometric shadows across architectural depth gaps.
* **Rule 4: Pass 2 Denoise Governs Illumination Integration:**
  * `0.35 – 0.45` (Feature Lock): Soft catalog fill.
  * `0.50 – 0.75` (Calibrated Range): Working zone for dramatic lighting. Synthesizes skin subsurface scattering, eye catchlights, and leather wrap while preserving likeness.
  * `0.75+` (Hallucination Boundary): May hallucinate physical softbox fixtures in the frame. Keep denoise $\le 0.75$.

---

## Production Recipes by Scenario

| Setup / Scenario | `specular_kick` | `chroma_bleed` | `filmic_white_point` | Pass 2 Denoise | Recommended Key Modifier |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **High-Fashion Beauty (Skin & Soft Fill)** | `0.40 – 0.50` | `0.40 – 0.45` | `3.5` | `0.50 – 0.55` | `Beauty Dish (Silver)` or `Octabox` |
| **Commercial Catalog (Mixed Fabrics & Leather)** | `0.40 – 0.50` | `0.35` | `3.5` | `0.55 – 0.60` | `Beauty Dish` or `Stripbox` |
| **Dark Tailored Apparel (Matte Wool / Velvet)** | `0.00 – 0.20` | `0.00` | `2.5` | `0.55 – 0.60` | `Octabox (Soft)` |
| **Cinematic Film Noir (High Chiaroscuro)** | `0.50 – 0.60` | `0.50` | `2.0 – 2.5` | `0.60 – 0.65` | `Fresnel Spotlight` or `Hard Snoot` |
| **Isolated Metallic / Chrome Props (Macro)** | `1.00 – 1.50` | `0.10` | `5.0 – 6.0` | `0.60 – 0.65` | `Direct Flash` or `Hard Snoot` |

---

## Installation

### 1. ComfyUI Manager (Recommended)
Search for `ComfyUI-StudioLight` in the ComfyUI Manager and install.

### 2. Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maciejtajber/ComfyUI-StudioLight.git
pip install -r ComfyUI-StudioLight/requirements.txt
```

### Dependencies & Models
* **Preprocessors:** Install `comfyui_controlnet_aux` for `DSINE-NormalMapPreprocessor` and `DepthAnythingV2Preprocessor`.
* **Models:** Standard `flux1-dev.safetensors` (or `flux1-krea-dev.safetensors`), CLIP text encoders, and VAE.

### Automated Test Suite
Verify math, photometric color spaces, and shader engines (runs standalone without ComfyUI):
```bash
python test_studiolight.py
```

---

## Master Workflows

* `workflows/StudioLight_Master.json`: Complete 2-Stage FLUX.1 Krea Dev production pipeline with prompt decoupling and metric shadow raymarching.
* `workflows/StudioLight_MethodA_Calibrated.json`: Fast portrait studio relighting workflow with interactive console gizmo and live VFX sphere probes.
* `workflows/StudioLight_MethodB_Trajectory.json`: Trajectory-driven camera-space lighting for animated sequences.

---

## Technical Documentation Index

| Document | Purpose |
| :--- | :--- |
| [`ARCHITECTURE_METRIC_3D_SHADOWS.md`](docs/ARCHITECTURE_METRIC_3D_SHADOWS.md) | Canonical metric raymarching, P0 occlusion gate, IGN dither derivation, stress benchmarks |
| [`ARCHITECTURE_PHYSICAL_AREA_EMITTERS.md`](docs/ARCHITECTURE_PHYSICAL_AREA_EMITTERS.md) | Geometric area apertures, 2D Hammersley sampling, radiometric normalization |
| [`ENGINE_WALKTHROUGH_AREA_EMITTERS.md`](docs/ENGINE_WALKTHROUGH_AREA_EMITTERS.md) | Vectorized PyTorch shader implementation, memory chunking, profiling |
| [`DISCOVERY_3D_METRIC_SHADOW_RAYMARCHING.md`](docs/DISCOVERY_3D_METRIC_SHADOW_RAYMARCHING.md) | Empirical discovery notes: 2D screen-space limits vs 3D camera-space |

---

## License

MIT License. Created by [Maciej Tajber](https://github.com/maciejtajber) // **METAMATERIAL** Applied VFX Systems.
