# Changelog

All notable changes to **ComfyUI-StudioLight** are documented in this file.

## [1.0] — 2026-11-04 — Initial Public Release

### Added
- **`StudioLightRig` (3-Point Photographic Console):**
  - Camera-space 3-point lighting rig (`Key`, `Fill`, `Rim`) with azimuth, elevation, stand distance in assumed scene meters ($r_{\text{ref}} = 1.8\text{m}$), Planckian locus Kelvin approximation ($2000\text{K}\text{–}10000\text{K}$), and 6 photographic studio modifiers (`Octabox (Soft)`, `Beauty Dish (Silver)`, `Stripbox`, `Fresnel Spotlight`, `Hard Snoot`, `Direct On-Camera Flash`).
  - **Depth-Reconstructed 3D Camera-Space Shadow Raymarching (`nodes/metric_shadow.py`):** Perspective-projected raymarching over depth-reconstructed 2.5D geometry with P0 front-of-ray occlusion gating ($D_{\text{scene}} \le D_{\text{ray}} - \epsilon$), Interleaved Gradient Noise (IGN) sub-step dither, and Turk (1990) uniform fan-triangle sampling across polygonal area apertures.
  - **3-Zone 3D Light Pointer (`Upper`, `Center`, `Lower`):** Curvature-guided subject auto-aim with multi-resolution $128\times128$ canonical finite differences and order-preserving depth relief compression (`depth_relief_scale`).
  - **Dual-Sphere VFX Probe & Studio Radar HUD:** Real-time 18% Matte Grey sphere, Mirror Chrome sphere with geometric modifier aperture reflections, and top-down 2D studio radar.
  - **Server-Side & Frontend Preset Synchronization:** 5 studio lighting presets (`Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, `Golden Hour Window`) resolved both server-side and in the ComfyUI browser UI.
- **`StudioLightConsistentComposite` (Schema 1 Scene-Linear Compositor):**
  - Dual relighting modes: **Add studio light** (preserves source plate radiance, practicals, and background ambient while adding direct diffuse, specular, and optional room fill) and **Replace subject light** (relights masked subject over estimated intrinsic albedo with controllable `source_preservation`).
  - Direct ~2-second cached lighting proxy preview output alongside labeled diagnostic contact sheets.
- **Bundled Intrinsic & Geometry Adapters:**
  - **`StudioLightMarigoldNormals` (`nodes/marigold_normals.py`):** Native float surface normal estimation (`prs-eth/marigold-normals-v1-1`) with automatic first-run Hugging Face caching.
  - **`StudioLightMarigoldDeLight` & `StudioLightDeLightStrength`:** Linear intrinsic albedo/shading/residual decomposition (`prs-eth/marigold-iid-lighting-v1-1`) and linear diffuse base blending.
  - **`StudioLightSubjectBackgroundBase` (`nodes/subject_background_base.py`):** Foreground subject mask integration (`BiRefNet-general`) with edge feathering.
- **`StudioLightSceneContract` & `SemanticSceneParser`:**
  - Deterministic clause-level semantic prompt parser preserving emotional markers, physiological details, and practical emitters while neutralizing conflicting directional lighting directives during Pass 2 refinement.
- **Production Workflows & Verification Suite:**
  - Two official production workflows: `workflows/StudioLight_Create.json` (draft-to-relight generative pipeline) and `workflows/StudioLight_Relight.json` (fixed-image relighting pipeline).
  - Automated unit, photometric contract, and workflow verification suite (`python3 -m unittest discover -s tests`).
