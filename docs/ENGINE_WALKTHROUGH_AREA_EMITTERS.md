# Walkthrough: Physically Motivated Geometric Area-Emitter Engine

The optical modifier architecture in ComfyUI-StudioLight has been upgraded from scalar heuristic approximations into a **physically motivated geometric area-emitter architecture**, following the 5 auditor directives.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 ENGINE UPGRADE: SCALAR SLIDERS -> AREA EMITTERS             │
└─────────────────────────────────────────────────────────────────────────────┘
  OLD: Scalar coefficients (wrap, power) ──> Generic isotropic circular blob
                                  │
                                  ▼
  NEW: 3D Calibrated Fixture Emitters (W, H, R) + 16 Area Samples + Ray-Plane Projection
       ├── Stripbox (30cm x 120cm)    ──> Tall rectangular bar (1:4 ratio)
       ├── Beauty Dish (55cm dish)     ──> Parabolic dish + center attenuation
       ├── Octabox (140cm octagon)     ──> Large soft octagonal disc
       ├── Fresnel Spotlight (30cm)    ──> Focused optical lens cutoff
       └── Hard Snoot (10cm tube)      ──> Tight pinpoint isolation
```

---

## Changes Made

### 1. Emitter Geometry & Profiles ([`nodes/color_utils.py`](../nodes/color_utils.py))
- Replaced scalar-only definitions with calibrated studio aperture dimensions:
  - **`Octabox (Soft)`:** $R=0.70\text{m}$ (140 cm diameter), $130^\circ$ broad Lambertian emission.
  - **`Stripbox`:** $W=0.30\text{m}, H=1.20\text{m}$ ($1:4$ ratio), $65^\circ$ narrow horizontal grid beam.
  - **`Beauty Dish (Silver)`:** $R=0.275\text{m}$ (55 cm diameter), deflector ratio $0.36$, transmission $0.12$.
  - **`Fresnel Spotlight`:** $R=0.15\text{m}$ (30 cm lens), $35^\circ$ focused optical cone.
  - **`Hard Snoot`:** $R=0.05\text{m}$ (10 cm pin-spot), $18^\circ$ tight cone.
  - **`Direct Flash`:** $W=0.08\text{m}, H=0.05\text{m}$ (compact strobe head).
- Preserved all legacy dictionary keys for **100% backward compatibility**.

### 2. Tangent Frame & Low-Discrepancy Sampling ([`nodes/shader_engine.py`](../nodes/shader_engine.py))
- **`generate_emitter_tangent_frame`:** Constructs an exact orthonormal basis $[\mathbf{u}_{\text{tangent}}, \mathbf{v}_{\text{tangent}}, \mathbf{w}_{\text{aim}}]$ for any fixture azimuth and elevation in 3D camera space.
- **`generate_emitter_samples_2d`:** Produces deterministic low-discrepancy 2D offsets in meters across the fixture face (Turk 1990 uniform polygon sampling for octagons, Hammersley 2D grid for rectangles, annular sampling for beauty dishes).

### 3. Vectorized $1/r^2$ Irradiance & Single-Layer Penumbra ([`nodes/shader_engine.py`](../nodes/shader_engine.py))
- **Fully Vectorized Area Integration:** Broadcasts $S=16$ sample positions against the surface pixels without Python iteration loops:
  $$\mathbf{P}_{\text{sample}} \in [S, 1, 1, 3] \implies [S, H, W, 3]$$
- **Geometric Inverse-Square $1/r^2$:** Normalized to reference stand distance $r_{\text{ref}} = 1.8\text{m}$:
  $$E(P) = \frac{1}{S} \sum_{k=1}^S (\mathbf{N} \cdot \mathbf{L}_k) \times (\mathbf{w}_{\text{aim}} \cdot -\mathbf{L}_k) \times \left(\frac{1.8}{d_k}\right)^2 \times V_k(P)$$
- **No Double-Softening:** Sample visibility uses sharp binary blocker ray-marching. Averaging visibility across the area emitter naturally forms contact-hardening penumbras without artificial post-blurring.

### 4. Specular Aperture Footprint Evaluator ([`nodes/shader_engine.py`](../nodes/shader_engine.py))
- Intersects reflection rays $\mathbf{R} = 2(\mathbf{N} \cdot \mathbf{V})\mathbf{N} - \mathbf{V}$ with the 3D fixture aperture plane.
- Evaluates the 2D aperture kernel with continuous edge feathering and microfacet roughness.
- Directly powers both:
  - **Chrome Mirror Ball Probe:** Produces distinct photographic fixture reflections.
  - **Subject Specular Pass:** Produces authentic corneal catchlights in the eyes and directional glints on skin, lips, and fabrics.

---

## Verification Results

The complete unit test suite and benchmark in [`test_studiolight.py`](../test_studiolight.py) ran and passed with 100% success:

| Test Item | Specification | Result |
|---|---|---|
| **Stripbox Emitter Geometry** | $1:4$ ratio, $|u| \le 0.15\text{m}, |v| \le 0.60\text{m}$ | **PASS** (Aspect = 4.00) |
| **Beauty Dish Sampling** | Annular bounds $0.099\text{m} \le r \le 0.275\text{m}$ | **PASS** (Zero center leaks) |
| **Stripbox Catchlight Reflection** | Anisotropic vertical elongation on mirror sphere | **PASS** (Aspect = 1.82 on curved sphere) |
| **Beauty Dish Parabolic Profile** | Continuous center attenuation with bright outer ring | **PASS** (Center = 0.00, Peak = 0.97) |
| **Fixture Scale Hierarchy** | Octabox area coverage $\gg$ Hard Snoot pin-spot | **PASS** (Octabox: 398 px vs Snoot: 3 px) |
| **Legacy Node Schema** | 26 required widgets + 5 optional (31 total) preserved | **PASS** (Zero schema regressions) |
| **End-to-End Suite Pipeline** | Stage 1 $\to$ Stage 2 $\to$ Bridge $\to$ Composite | **PASS** (Full execution verified) |

All 22 test modules passed cleanly!
