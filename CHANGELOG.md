# Changelog

All notable changes to **ComfyUI-StudioLight** are documented in this file.

## [Unreleased]

- Add explicit scene-linear exposure and luminance-based HDR tone mapping with display gamut compression; preserve legacy defaults for existing workflows.
- Add opt-in relative-depth background protection and enable it in the master example. Far backgrounds are protected artistically, not classified as sky or invalid geometry.
- Add eight CPU/CUDA correction regressions and safe float32 replay loading for fixed-draft visual comparisons.

- Preserve scene-linear HDR lighting and specular radiance until compositing.
- Wire prompt sanitizer into the master workflow without changing sampler settings.
- Reject incompatible geometry/compositor batches; retain matched and singleton broadcast behavior.
- Add release regression checks and an automated CUDA/ComfyUI visual comparison harness.

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
