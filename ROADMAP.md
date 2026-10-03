# ComfyUI-StudioLight — Roadmap & Strategic Frontiers

> **Strategic Architecture & Next-Gen Development Plan**  
> **Current Status:** v0.1 Production Calibrated (Tuesday, Oct 6, 2026 Drop)  
> **Core Mission:** Direct photographic studio illumination across generative diffusion models with deterministic spatial control.

---

## 🏆 Shipped in v0.1 (Calibrated Prototype)

The following foundational systems are completed, calibrated on FLUX Krea Dev, and verified by 15/15 automated test suites:

- [x] **Method A (2-Stage Calibrated Relighting):** The production workhorse. Combines neutral flat albedo draft passes with deferred 3D lit proxy plates and stochastic re-diffusion (Steps 10–30).
- [x] **3D Pinhole Camera Back-Projection:** Full metric OpenGL camera space ($40^\circ$ FOV portrait lens) mapping relative depth to $Z = -1.5\text{m}$ (nose) down to $-4.5\text{m}$ (background), with non-square aspect ratio ($W/H$) correction.
- [x] **Calibrated Photographic Power Falloff & Hermite Background Taper:** Per-light Euclidean stand distance falloff $(1/d)^{1.6}$ (sub-quadratic to approximate practical studio bounce) with smooth cubic Hermite distance tapering ($d_{\text{norm}} < 0.20$), preventing studio light bleeds onto distant backgrounds.
- [x] **Optical Forward Modifier Beam Coning & Horizon Gating:** Directed beam gating that stops backwards light leakage ($\cos\theta_{\text{beam}} > 0$) and specular horizon gating ($N \cdot L > 0$), completely eliminating architectural edge halos.
- [x] **Automatic Albedo Prompt Decoupling (`StudioPromptSanitizer`):** Strips high-gloss/specular terms from Pass 1 while keeping 100% of the master prompt in Pass 2 to eliminate pre-baked specular blobs.
- [x] **Photometric Composite & Rational Shoulder:** Zero double-flooring in linear Rec.709 space. Quadratic rational shoulder tonemapping with active `white_point` scaling and dielectric Fresnel base gating ($F_0 \ge 0.12$).
- [x] **Dedicated Specular Pass (4th Output):** Razor-sharp Blinn-Phong dielectric catchlights (2.8% pixel area) with dedicated `specular_pass` output and composite hook.
- [x] **DSINE / OpenGL Coordinate Parity & Depth Polarity:** Verified $+Y$ OpenGL coordinate convention, automatic stage asset detection, `invert_normals_x`, and `invert_depth` toggles.
- [x] **Frozen 27-Widget Required Schema:** 100% backward compatibility for all existing user workflows with zero slider-shift risk.
- [x] **Curated Studio Presets:** `Rembrandt Classic`, `High-Fashion Beauty`, `Direct On-Camera Flash`, `Cinematic Film Noir`, and `Golden Hour Window`.

---

## 🗺️ Product Roadmap: Four Focused Phases

```text
v0.1  ✅ PROOF (Shipped)
3D geometry → physical lighting → stochastic FLUX relighting

v0.2  🎯 PRODUCTIZATION (Immediate Priority)
1. Optical penumbra & soft-shadow engine
2. Native 3D pose + camera studio
3. GLB/OBJ 3D asset import
4. Universal 2D image relighting (Img2Img)

v0.3  🧪 ADVANCED
5. Material-aware lighting (SAM-guided)
6. Geometry & identity locking
7. Automated reference-pose → 3D pose

v0.4  🎥 VIDEO
8. Temporal lighting coherence
9. Wan 2.1 & LTX-Video integration

R&D   🔬 CONTINUOUS LATENT INJECTION
10. Continuous flow trajectory injection (Method B)
```

---

## 🎯 v0.2: Productization & Core Experience

### 1. Optical Penumbra & Soft Shadow Engine (Immediate Priority)
* **The Problem:** Current screen-space raymarching simulates an infinitesimal point light, producing hard shadow silhouettes regardless of modifier choice.
* **The Solution:** Implement modifier aperture modeling:
  $$\text{Penumbra Blur Radius} \propto \frac{\text{Modifier Diameter} \times (Z_{\text{background}} - Z_{\text{subject}})}{d_{\text{light}}}$$
* A 150cm Octabox close to a subject casts a wide, creamy penumbra onto background walls, while a snoot or direct flash preserves crisp, hard micro-shadows.

### 2. Native 3D Pose + Camera Studio & GLB/OBJ Import
* **The Goal:** Eliminate external web dependencies (e.g. PoseMy.Art).
* **The Implementation:**
  * Embed a dedicated interactive Three.js 3D viewport directly on the ComfyUI canvas.
  * Direct FK/IK bone manipulation on mannequins + interactive orbital camera positioning.
  * Drag-and-drop import for custom 3D assets (`.glb`, `.gltf`, `.obj`).

### 3. Universal 2D Image Relighting (Img2Img Pipeline)
* **The Goal:** Relight pre-existing client photography, catalog assets, and historical portraits.
* **The Flow:**
  ```text
  Existing Photo
  → Depth (Depth Anything V2) + Normals (DSINE) + Intrinsic Albedo
  → StudioLightRig (3D lights & modifiers)
  → FLUX Krea Dev / SDXL (Denoise 0.50–0.70)
  → Photometrically Relit Output
  ```

---

## 🧪 v0.3: Advanced Controls

* **Material-Aware Lighting:** Automated semantic material separation (MobileSAM / SAM 3) to route diffuse vs. high-dielectric specular kicks (e.g. matte wool vs. patent leather) without manual masking.
* **Identity & Geometry Locking:** Depth/Normal ControlNet stabilization at high chiaroscuro denoise ($> 0.70$).
* **Reference-Pose to 3D Alignment:** Automatically fit 3D mannequin joints to input reference imagery (DWPose / OpenPose mapping).

---

## 🎥 v0.4: Video & Temporal Coherence

* **Keyframeable 3D Light Vectors:** Animate Key/Fill/Rim angles and intensities across video frame sequences.
* **Temporal Photometric Shading:** Consistent 3D illumination for Wan 2.1 and LTX-Video, eliminating frame-to-frame prompt lighting flicker.

---

## 🔬 R&D: Continuous Flow Trajectory (Method B)

* **Premise:** Single-pass continuous diffusion injection intercepting clean $x_0$ at flow step $\sigma_{10} = 0.863338$ via native Euler callback.
* **Current Status:** Mathematical invariants verified ($\le 4.77 \times 10^{-7}$ error). Empirical tests show deterministic delta injection freezes latents (~2.5% luminance shift).
* **Next Exploration:** Adaptive noise re-injection at $\sigma_{10}$ to allow flow vectors to repaint shadow penumbras without destabilizing facial structure. Kept in R&D to avoid blocking core product execution.
