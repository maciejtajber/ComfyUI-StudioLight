# Roadmap: ComfyUI-StudioLight

StudioLight bridges deterministic 3D photographic lighting with generative diffusion models. Development focuses on physical realism, workflow ergonomics, and expanding model architectures.

---

### Core Foundations (Shipped) ✅
* [x] **3-Point Studio Rig:** Independent Key, Fill, and Rim positioning in 3D camera space (Azimuth, Elevation, Distance).
* [x] **Photometric Shading Engine:** Linear Rec.709 compositing, Blackbody Kelvin color balance (2000K–10000K), and optical forward modifier beam cones.
* [x] **Calibrated Power Falloff:** $1/d^{1.6}$ distance falloff paired with a Hermite background distance taper ($d_{\text{norm}} < 0.20$) to prevent background architecture bleed.
* [x] **Dedicated Specular Catchlights:** Isolated Blinn-Phong specular lobes (`specular_pass`) composited through a $C^1$-continuous rational shoulder tonemapper with active headroom control.
* [x] **Optical Penumbra & Area Soft Shadows:** Screen-Space PCSS with Vogel disk sampling simulating modifier diameters (Octabox 140cm vs. Snoot 10cm) with contact hardening.
* [x] **Automated Test Coverage:** 16 standalone unit test suites verifying math, PCSS soft shadows, schema parity, and color conversions.

---

### Upcoming Milestones 🎯
* [ ] **Temporal & Video Diffusion Integration:** Model-agnostic integration piping 32-bit linear photometric relighting plates into video diffusion attention streams and spatio-temporal latents (Wan 2.1, LTX-Video, Hunyuan).
* [ ] **Temporal Light Vector Keyframing:** Continuous interpolation of light coordinates, intensity, and Kelvin temperatures across video frame sequences for smooth cinematic sweeps.
* [ ] **Expanded Model Architectures (Krea 2 & Beyond):** Conditioning adapters and native CFG response dynamics for newly released foundation models.
* [ ] **Universal 2D Image Relighting (Img2Img):** Standalone relighting pipeline accepting existing 2D photographs or renders via intrinsic albedo estimation and depth/normal extraction.
* [ ] **`StudioModifierTuner` Companion Node:** Dedicated utility node enabling continuous slider control over physical aperture diameter (0.01m – 3.00m), asymmetric modifier aspect ratios (stripbox width vs. length), and internal baffle diffusion softness.
* [ ] **Native 3D Stage & Asset Ingestion:** Integrated lightweight Three.js/WebGL canvas supporting `.glb` / `.obj` positioning, camera alignment, and interactive light gizmos inside ComfyUI.
* [ ] **Live Interactive Radar Overhaul:** Enhanced top-down radar console with multi-light distance ring manipulation and interactive beam-cone angles.
* [ ] **Material-Aware Specular Routing:** Automatic dielectric vs. metallic reflectance handling to prevent edge solarization across high-contrast fabric combinations.
* [ ] **Reference Pose-to-Lighting Matching:** Extract incident light directions from reference studio photographs and auto-populate the 3-point console coordinates.

---

### Active Research & Exploration (R&D) 🔬
* **Continuous Latent Trajectory Injection (Method B):** Non-destructive Euler ODE delta injection ($\Delta x = (x_0' - x_0) \cdot \text{strength}$) at flow step $\sigma_{10} = 0.863338$, exploring adaptive noise scaling to achieve material re-synthesis without stochastic restarts.
