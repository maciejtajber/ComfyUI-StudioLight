# Roadmap: ComfyUI-StudioLight

StudioLight bridges deterministic 3D photographic lighting with generative diffusion models. Development follows a phased release structure focused on physical realism, workflow ergonomics, and production reliability.

---

### Phase 1: v0.1 — Foundations & Proof (Shipped) ✅
* [x] **3-Point Studio Rig:** Independent Key, Fill, and Rim positioning in 3D camera space (Azimuth, Elevation, Distance).
* [x] **Photometric Shading Engine:** Linear Rec.709 compositing, Blackbody Kelvin color balance (2000K–10000K), and optical forward modifier beam cones.
* [x] **Calibrated Power Falloff:** $1/d^{1.6}$ distance falloff paired with a Hermite background distance taper ($d_{\text{norm}} < 0.20$) to prevent background architecture bleed.
* [x] **Dedicated Specular Catchlights:** Isolated Blinn-Phong specular lobes (`specular_pass`) composited through a $C^1$-continuous rational shoulder tonemapper with active headroom control.
* [x] **2-Stage Midstream Pipeline (Method A):** Geometry draft extraction (Step 10 normals & depth) coupled with stochastic re-diffusion (denoise 0.50–0.75).
* [x] **Automated Test Coverage:** Standalone unit test suites verifying math, schema parity, and color conversions.

---

### Phase 2: v0.2 — Productization & Physical Realism 🎯
* [ ] **Optical Penumbra & Area Soft Shadows:** Depth-dependent shadow raymarching simulating physical modifier diameters (Octabox 150cm vs. Snoot 10cm) for realistic umbra/penumbra falloff.
* [ ] **Universal 2D Image Relighting (Img2Img):** Standalone relighting pipeline accepting existing 2D photographs or renders via intrinsic albedo estimation and depth/normal extraction.
* [ ] **Native 3D Stage & Asset Ingestion:** Integrated lightweight Three.js/WebGL canvas supporting `.glb` / `.obj` positioning, camera alignment, and interactive light gizmos inside ComfyUI.
* [ ] **Live Interactive Radar Overhaul:** Enhanced top-down radar console with multi-light distance ring manipulation and interactive beam-cone angles.

---

### Phase 3: v0.3 — Advanced Material Control 🧪
* [ ] **Material-Aware Specular Routing:** Automatic dielectric vs. metallic reflectance handling to prevent edge solarization across high-contrast fabric combinations.
* [ ] **Geometric Identity Anchoring:** Targeted high-frequency edge stabilization during extreme chiaroscuro transitions.
* [ ] **Reference Pose-to-Lighting Matching:** Extract incident light directions from reference studio photographs and auto-populate the 3-point console coordinates.

---

### Phase 4: v0.4 — Temporal & Video Integration 🎥
* [ ] **Temporal Vector Keyframing:** Continuous interpolation of light coordinates, intensity, and Kelvin temperatures across video frame sequences.
* [ ] **Video Diffusion Pipeline (Wan 2.1 / LTX-Video):** Model-agnostic integration piping 32-bit linear photometric relighting plates into video diffusion attention streams.

---

### Active Research & Exploration (R&D) 🔬
* **Continuous Latent Trajectory Injection (Method B):** Non-destructive Euler ODE delta injection ($\Delta x = (x_0' - x_0) \cdot \text{strength}$) at flow step $\sigma_{10} = 0.863338$, exploring adaptive noise scaling to achieve material re-synthesis without stochastic restarts.
