# ComfyUI-StudioLight — Roadmap & Strategic Frontiers

> **Strategic Architecture & Next-Gen Development Plan**  
> **Current Status:** v0.1 Production Calibrated (Tuesday, Oct 6, 2026 Drop)  
> **Core Mission:** Eliminate lighting RNG in generative diffusion through rigorous physical optics and 3D camera-space simulation.

---

## 🏆 Shipped in v0.1 (Production Calibrated)

The following foundational systems are completed, calibrated on FLUX Krea Dev, and verified by 15/15 automated test suites:

- [x] **Method A (2-Stage Calibrated Relighting):** The production workhorse. Combines neutral flat albedo draft passes with deferred 3D lit proxy plates and stochastic re-diffusion.
- [x] **Method B (Continuous Trajectory Injection):** Intercepts un-overshot clean prediction $x_0$ at flow step $\sigma_{10} = 0.863338$ and algebraically injects the lighting delta $\Delta x = (x_0' - x_0) \cdot \text{strength}$ directly into the diffusion trajectory.
- [x] **3D Pinhole Camera Back-Projection:** Full metric OpenGL camera space ($40^\circ$ FOV portrait lens) mapping relative depth to $Z = -1.5\text{m}$ (nose) down to $-4.5\text{m}$ (background), with non-square aspect ratio ($W/H$) correction.
- [x] **Photographic Falloff & Hermite Background Taper:** Per-light Euclidean stand distance falloff $(1/d)^{1.6}$ with smooth cubic Hermite distance tapering ($d_{\text{norm}} < 0.20$), preventing studio light bleeds onto distant backgrounds.
- [x] **Optical Forward Modifier Beam Coning:** Directed beam gating that stops backwards light leakage, completely eliminating architectural edge halos.
- [x] **Automatic Albedo Prompt Decoupling (`StudioPromptSanitizer`):** Strips high-gloss/specular terms from Pass 1 while keeping 100% of the master prompt in Pass 2 to eliminate pre-baked specular blobs.
- [x] **Photometric Composite & Rational Shoulder:** Zero double-flooring in linear Rec.709 space. Quadratic rational shoulder tonemapping with active `white_point` scaling and dielectric Fresnel base gating ($F_0 \ge 0.12$).
- [x] **Dedicated Specular Pass (4th Output):** Razor-sharp Blinn-Phong dielectric catchlights (2.8% pixel area) with dedicated `specular_pass` output and composite hook.
- [x] **DSINE / OpenGL Coordinate Parity & Depth Polarity:** Verified $+Y$ OpenGL coordinate convention, automatic stage asset detection, `invert_normals_x`, and `invert_depth` toggles.
- [x] **Frozen 27-Widget Required Schema:** 100% backward compatibility for all existing user workflows with zero slider-shift risk.
- [x] **Curated Studio Presets:** `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.

---

## 🚀 Phase 2: Next Strategic Frontiers (v0.2 & Beyond)

With core 3-point lighting physics proven and operational, the following five frontiers represent the highest-leverage engineering targets:

### 1. Optical Penumbra & Soft Shadow Raymarching (Immediate Priority)
* **The Problem:** Current screen-space shadow raymarching simulates an infinitesimal mathematical point light, casting razor-sharp shadow edges (e.g. sharp shadow silhouettes on background columns).
* **The Solution:** Implement physical modifier aperture modeling:
  $$\text{Penumbra Blur Radius} \propto \frac{\text{Modifier Diameter} \times (Z_{\text{background}} - Z_{\text{subject}})}{d_{\text{light}}}$$
* A 150cm Octabox close to a subject will cast a wide, creamy diffused penumbra onto background walls, while a snoot or direct flash casts a crisp, sharp shadow edge.

### 2. Universal Image-to-Image Relighting Pipeline
* **The Problem:** Currently, the calibrated pipeline generates both the base image (Pass 1) and the lighting pass (Pass 2). Users need to relight pre-existing client photos, historical portraits, and catalog assets.
* **The Solution:**
  * **Input Node:** Single-click loading of any arbitrary 2D image.
  * **Intrinsic Image Decomposition:** Extract depth (Depth Anything V2) and normals (DSINE), followed by an albedo estimation pass to strip existing hard shadows before feeding into `StudioLightRig`.
  * **Preservation Masking:** Re-diffuse lighting onto skin, hair, and clothing while keeping original eyes, teeth, and background geometry 100% untouched.

### 3. Segment Anything 3 (SAM 3) Semantic Material Gating
* **The Problem:** Different materials require different lighting settings. Black wool requires `specular_kick = 0.00` to prevent thermal edge solarization, while patent leather bags and jewelry demand `specular_kick = 2.5+`.
* **The Solution:**
  * Integrate lightweight zero-shot segmentation (SAM 3 / MobileSAM) to create semantic material masks.
  * Auto-route materials inside `StudioPhotometricComposite`:
    * `Mask: Skin` $\to$ Subsurface scattering + soft specular ($1.0$).
    * `Mask: Dark Wool / Matte Fabric` $\to$ Pure diffuse absorption ($0.0$ specular kick).
    * `Mask: Leather / Metal / Eyes` $\to$ High dielectric/conductor specular boost ($2.5+$).

### 4. Dual Depth + Normal ControlNet Geometry Locking
* **The Problem:** Pushing Pass 2 denoise to high chiaroscuro levels ($0.70 – 0.80$) gives extreme light transformation but introduces subtle facial identity drift.
* **The Solution:**
  * Feed the Pass 1 depth and normal maps into a low-weight ($0.30 – 0.40$) ControlNet Union during Pass 2.
  * Locks exact bone structure, jawline, and nose contours with zero drift, granting FLUX unlimited freedom to repaint illumination gradients.

### 5. Temporal Video Lighting Studio (Wan 2.1 & LTX-Video)
* **The Problem:** AI video models suffer from flickering lighting across frames when prompt words shift.
* **The Solution:**
  * Animate 3D light vectors across a timeline (e.g. keyframing a Key Light moving from $-45^\circ$ to $+45^\circ$ across 48 frames).
  * Compute temporally coherent 3D lighting maps across video latents, producing authentic camera pans and rotating headlights across moving subjects.

---

## 📅 Roadmap Timeline

| Milestone | Target Date | Scope |
|---|---|---|
| **v0.1 Public Drop** | **Tuesday, Oct 6, 2026** | Step 10 Midstream 3D Extraction, Method A Calibrated, 15/15 Tests, Hero Renders, Manager Index |
| **v0.1.1 Optical Penumbra** | **Mid-October 2026** | Area-light soft shadow penumbra in `shader_engine.py` |
| **v0.2 Img2Img Relighting** | **Late October 2026** | Intrinsic albedo stripping + arbitrary 2D photo relighting |
| **v0.3 Material Gating** | **November 2026** | SAM 3 semantic material masking for automatic wool vs. leather handling |
| **v1.0 Video Studio** | **Q4 2026** | Animated temporal light paths for Wan 2.1 and LTX-Video |
