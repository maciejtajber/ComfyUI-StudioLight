# Architecture Plan: StudioLight Physically Based Geometric Area-Emitter Architecture

## Goal Description
Transform ComfyUI-StudioLight's modifier system from heuristic scalar coefficients into a **physically based geometric area-emitter architecture**. Every studio modifier becomes a defined virtual emitter with 3D metric dimensions, emission distribution, fully vectorized multi-sample area integration, single-layer penumbra visibility, and an analytical aperture reflection in reflection space.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│             PHYSICALLY BASED GEOMETRIC AREA-EMITTER ARCHITECTURE            │
└─────────────────────────────────────────────────────────────────────────────┘
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
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      ▼
                         AGGREGATED STUDIO COMPOSITE
           • Calibrated Studio Exposure applied as independent layer
           • 3D Bust: Contact-hardening soft shadows + authentic catchlights
           • Chrome Ball: True photographic fixture shapes (no generic blobs)
           • Backward Compatibility: 100% locked schema (zero broken workflows)
```

---

## The 5 Core Architectural Revisions

### 1. No Double-Softening (Area Sampling Alone Creates Penumbra)
- **Problem:** Evaluating PCSS / Vogel disk filtering on top of multi-sample area lighting creates muddy, double-blurred shadows.
- **Solution:** Each emitter sample $k \in [1 \dots S]$ behaves as a point source evaluating sharp geometric screen-space visibility $V_k \in \{0, 1\}$.
- **Result:** Averaging the $S$ visibility terms naturally produces contact-hardening penumbras (knife-sharp contact under the chin/collar, progressively softening with blocker distance) without artificial post-blurring.

### 2. Fully Vectorized Sample Tensor `[S, H, W, 3]`
- **No Python sample loops.** Emitter sample coordinates $\mathbf{P}_{\text{sample}} \in [S, 1, 1, 3]$ broadcast against surface points $\mathbf{P}_{\text{surf}} \in [1, H, W, 3]$.
- PyTorch evaluates all $S=16$ directions, inverse-square distances, receiver cosines, and emitter cosines in parallel batch operations.
- Benchmark target: $<50\text{ ms}$ on standard hardware.

### 3. True Physical $1/r^2$ Separated from Studio Exposure
- **Raw Physical Irradiance:**
  $$E_{\text{raw}}(P) = \frac{1}{S} \sum_{k=1}^S \frac{\max(\mathbf{N} \cdot \mathbf{L}_k, 0) \times \max(\mathbf{L}_{\text{aim}} \cdot (-\mathbf{L}_k), 0)}{r_k^2} V_k(P)$$
- **Studio Exposure / Calibration:**
  $$E_{\text{calibrated}}(P) = E_{\text{raw}}(P) \times r_{\text{ref}}^2 \times \text{Intensity} \times C_{\text{studio}}$$
  where $r_{\text{ref}} = 1.8\text{m}$ standard portrait stand reference distance.
  This grounds the engine in rigorous physics while preserving intuitive studio photography exposure control.

### 4. Continuous Source Profiles (No Hardcoded Stereotypes)
- **Beauty Dish:** Modeled as a parabolic reflector with central deflector disk. It is not an arbitrary zero-center boolean hole; rather, the deflector causes an analytical forward dip in irradiance, while the parabolic silver/white bowl maintains interior fill.
- **Fresnel / Snoot / Octabox:** Governed by geometric boundary distance functions and smooth cosine beam emission profiles $\cos^\gamma(\theta_e)$.

### 5. Decoupled Modular Engine Structure
The codebase cleanly separates:
```
nodes/
  ├── color_utils.py       # Emitter physical dimensions & profiles
  └── shader_engine.py
        ├── Emitter Frame & Sampling       (Geometry & tangent frames)
        ├── Diffuse Emission Integrator   (Vectorized 1/r² irradiance)
        ├── Visibility Evaluator           (Multi-sample blocker tests)
        └── Specular Aperture Evaluator    (Ray-plane reflection profiles)
```

---

## Implementation Order
1. **Emitter Geometry:** Define metric profiles in `color_utils.py`.
2. **Vectorized Sampling:** Build `generate_emitter_samples_3d` in `shader_engine.py`.
3. **Physical $1/r^2$ Irradiance:** Implement vectorized tensor broadcasting.
4. **Per-Sample Visibility:** Screen-space blocker tests without post-blur.
5. **Aggregation & Exposure:** Combine diffuse + shadow with reference normalization.
6. **Specular Aperture Footprint:** Ray-plane reflection space evaluator for chrome ball & bust.
7. **Automated Verification:** Unit tests & performance benchmark in `test_studiolight.py`.
