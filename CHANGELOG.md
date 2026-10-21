# Changelog

All notable changes to **ComfyUI-StudioLight** are documented in this file.

## [2.0.0] — 2026-10-21 — Semantic Prompt & Albedo Architecture v2

### Added
- **`SemanticSceneParser` (`nodes/semantic_parser.py`):**
  - High-precision deterministic clause-level semantic prompt parser decomposing arbitrary master prompts into structured `SceneDescription` data models.
  - **Zero Content Destruction:** 100% preservation of emotional markers (tears, crying), physiological details (sweat, wet skin), physical emitter props (smartphones, candles, torches, lanterns, neon signs), clothing, accessories, and stylistic descriptors.
  - **Zero Orphan Grammar:** Complete elimination of dangling prepositions, conjunctions, and orphan adjectives (e.g. `illuminated by a harsh`, `deep casting on`).
  - **Material-Aware Classification:** Automatic categorization across `DIFFUSE`, `DIELECTRIC_GLOSSY`, `METALLIC`, `TRANSPARENT`, `TRANSLUCENT`, and `EMISSIVE` with reflective surface preservation.
- **Lighting Policy Resolver (`RIG_OVERRIDE` vs `HYBRID_PRACTICAL`):**
  - **`RIG_OVERRIDE` (Default):** The StudioLight 3D physical rig is the absolute authority for directional illumination. Conflicting prompt directives (flashes, spotlights, rim lights, cast shadows, chiaroscuro) are cleanly neutralized from both Pass 1 and Pass 2 prompts.
  - **Pass 2 Concatenation Decoupling:** Terminated the automatic injection of the 80-word generated studio rig narrative into Pass 2 img2img (`StringConcatenate` in Node 21 decoupled from generation). Pass 2 receives the clean scene description + minimal refinement tag (`"preserving subject identity, composition, material characteristics, and fine surface details"`), allowing the 3D lit proxy latent to dictate 100% of spatial lighting with zero text-induced hallucinations.
  - **`HYBRID_PRACTICAL`:** Rig controls directional key/fill/rim while in-scene practical emitters retain their intended emissive glow descriptions in Pass 2.
  - **`VERBATIM_RAW`:** Backward-compatibility option returning untouched master prompt for Pass 2 if explicitly requested.
- **Neutral Diffuse Reference Capture Guidance:**
  - Replaced fictional "Log-C" buzzwords with photographically grounded neutral capture guidance: `"neutral diffuse reference capture, broad balanced illumination, preserved surface colors and textures, shadowless base exposure, recoverable detail in dark regions, no strong color cast"`.
- **Workflow & Graph Synchronization (`workflows/StudioLight_Master.json`):**
  - Pass 2 `CLIPTextEncode` directly receives `pass2_lit_prompt` from Node 8 (`StudioPromptSanitizer`).
  - `StudioLightRig` `lighting_prompt` routed to Node 22 (`PreviewAny`) for real-time diagnostic telemetry / visual inspection.
- **Comprehensive Verification Suite (`test_prompt_architecture_v2.py`):**
  - Full automated coverage for mandatory benchmark prompts (Crying woman with phone, Chrome android, Handbag on marble, Techno club flash, Monk with candle).
  - 103/103 test gates passed across `test_prompt_architecture_v2.py`, `test_studiolight.py`, `test_extended_3domain.py`, and `python3 -m unittest`.

## [1.0.0] — 2026-10-12

### Added
- **`StudioLightRig` (3-Point Photographic Console):**
  - Camera-space 3-point lighting rig (`Key`, `Fill`, `Rim`) with azimuth, elevation, stand distance in meters ($r_{\text{ref}} = 1.8\text{m}$), blackbody Kelvin temperature ($2000\text{K}\text{–}10000\text{K}$), and 6 photographic studio modifiers (`Octabox (Soft)`, `Beauty Dish (Silver)`, `Stripbox`, `Fresnel Spotlight`, `Hard Snoot`, `Direct On-Camera Flash`).
  - **Depth-Reconstructed 3D Camera-Space Shadow Raymarching (`nodes/metric_shadow.py`):** Perspective-projected raymarching over depth-reconstructed 2.5D geometry with P0 front-of-ray occlusion gating ($D_{\text{scene}} \le D_{\text{ray}} - \epsilon$), Interleaved Gradient Noise (IGN) sub-step dither, and Turk (1990) uniform fan-triangle sampling across polygonal area apertures.
  - **3-Zone 3D Light Pointer (`Upper`, `Center`, `Lower`):** Curvature-guided portrait face auto-aim with multi-resolution $128\times128$ canonical finite differences and order-preserving depth relief compression (`depth_relief_scale`).
  - **Dual-Sphere VFX Probe & Studio Radar HUD:** Real-time 18% Matte Grey sphere, Mirror Chrome sphere with geometric modifier aperture reflections, and top-down 2D studio radar.
  - **Server-Side & Frontend Preset Synchronization:** 5 studio lighting presets (`Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, `Golden Hour Window`) resolved both server-side and in the ComfyUI browser UI.
- **`StudioPhotometricComposite` (Scene-Linear Rec.709 Compositor):**
  - Linear Rec.709 (`sRGB -> Linear -> sRGB`) diffuse and isolated specular lobe integration (`specular_kick`), midtone chromatic bleed (`chroma_bleed`), and rational filmic shoulder tonemapping (`filmic_white_point`).
- **`SIShadowHighlight` (Latent Preparation Darkroom):**
  - Bilateral Hermite cubic spline ($3t^2 - 2t^3$) shadow recovery and highlight compression with zero midtone bleed, plus bounded sigmoidal midtone contrast.
- **`StudioPromptSanitizer` (Pass 1 Albedo Decoupler):**
  - Automated prompt splitter stripping high-specular and directional tokens from Pass 1 draft generation while routing the full master prompt to Pass 2.
- **Verification & Documentation Suite:**
  - 22-test mathematical, geometric, and workflow schema verification suite (`test_studiolight.py`; device-parity tests run where CUDA / Apple Silicon MPS is available, while GitHub Actions CI covers CPU).
  - Complete 2-stage production workflow (`workflows/StudioLight_Master.json`) and 4 technical architecture documents in `docs/`.


## Unreleased — Photometric transport and diagnostics (2026-10-15)

- Add explicit rig telemetry without moving existing output slots or changing the four-output Python rig API.
- Separate direct diffuse, raw specular and ambient; make strobe-off background modulation consistent.
- Add opt-in RGB room-fill heuristic, bounded user-masked material reconstruction, and low-signal chroma gate. No fabricated metric enclosure or automatic warm-skin bias.
- Add labelled sRGB contact sheet, HDR luminance/gamut mapping, strict tensor/scalar validation and per-image batch handling.
- Restore legacy exposure, background-protection and Soft Light controls; preserve haze, crevice correction and cyclorama safeguards.
- Wire telemetry/depth/preview in shipped workflows; add photometric contract regression gates and prevent telemetry bypassing the HDR clamp experiment.
