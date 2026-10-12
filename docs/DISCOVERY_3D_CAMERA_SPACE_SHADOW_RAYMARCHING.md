# Architectural Discovery: 2D Screen-Space Shadows vs. 3D Camera-Space Raymarching

**Date:** October 8, 2026  
**Module:** `nodes/metric_shadow.py` (`screen_space_directional_shadow_metric`) · `nodes/shader_engine.py`  
**Status:** Resolved & Deployed in v1.0.0 (`nodes/metric_shadow.py` replaces the legacy 2D screen-space stepper)

---

## 1. Empirical Observation (The Problem)

During calibrated live FLUX studio runs on full-body architectural plates (model standing in front of neoclassical columns and stone walls):
* **Contact shadows:** Look crisp under chin, collar, and lapels.
* **Cast shadows on background architecture:** Fail to behave like true 3D perspective shadows. Instead of projecting across the 3D camera-space gap $\Delta Z$ and wrapping across the cylindrical fluting of the columns, the shadow behaves like a **2D drop shadow / contact halo** tightly clinging to the coat silhouette.

---

## 2. Mathematical Root Cause Analysis

Inspection of [`nodes/shader_engine.py`](../nodes/shader_engine.py) reveals the exact cause:

```python
dx = (lx / norm_2d) * (max_dist / steps)
dy = (ly / norm_2d) * (max_dist / steps)

for s in range(1, steps + 1):
    offset_x = dx * s * 2.0
    offset_y = dy * s * 2.0
    # ...
    sampled_d = F.grid_sample(...)
    depth_diff = sampled_d - d_in
    is_closer = depth_diff > bias
    weight = 1.0 - (s / (steps + 1))
    shadow = shadow - occlusion * 0.25
```

### Why this fails 3D geometry:
1. **Marches in Flat 2D Pixel Space $(u, v)$:** The ray steps fixed screen fractions (`max_dist = 0.08`, ~40 pixels). It **never advances along the 3D light ray vector $L_z$** in camera space.
2. **Zero Depth Gap ($\Delta Z$) Awareness:** Whether a background column is $0.5\text{m}$ or $8.0\text{m}$ behind the model, the algorithm searches the **identical 40 pixels on screen**. It cannot project across distance.
3. **Artificial Screen Fade (`weight = 1.0 - s / steps`):** The further the 2D ray marches, the fainter the shadow becomes, turning it into an ambient occlusion / vignette edge rather than a solid cast shadow.
4. **No Surface Wrapping:** Because it doesn't intersect 3D geometry, the shadow cannot deform over the curved 3D geometry of fluted columns, stairs, or walls.

---

## 3. Mathematical Solution: 3D Camera-Space Raymarching

StudioLight already reconstructs the complete 3D camera-space coordinate field:
$$\mathbf{P}_{\text{surf}} = (X, Y, Z) \in \mathbb{R}^3$$
using monocular depth calibrated to studio scale and camera intrinsics ($FOV = 40^\circ$).

### The 3D Raymarching Algorithm:
1. For any receiver point $\mathbf{P}_{\text{surf}} = (X_r, Y_r, Z_r)$, cast a ray toward the light source:
   $$\mathbf{P}(t) = \mathbf{P}_{\text{surf}} + t \cdot \mathbf{L}, \quad t \in [0, d_{\text{max}}]$$
2. At each 3D step $t$, project $(X(t), Y(t), Z(t))$ back into screen coordinates $(u(t), v(t))$:
   $$u(t) = \frac{X(t)}{-Z(t) \cdot \tan(\text{FOV}/2) \cdot \text{aspect}}, \quad v(t) = \frac{Y(t)}{-Z(t) \cdot \tan(\text{FOV}/2)}$$
3. Sample the scene depth map at $(u(t), v(t))$: $Z_{\text{scene}}(u(t), v(t))$.
4. If $Z(t) \le Z_{\text{scene}}(u(t), v(t))$ (the ray has dipped behind a surface occluder):
   $$\text{Occlusion} = 1.0 \implies \mathbf{P}_{\text{surf}} \text{ is in shadow}$$

### Outcomes:
* **True Distance Projection:** A wall $4\text{m}$ behind the model catches the shadow meters away, exactly matching the 3D light vector.
* **Authentic Geometric Wrapping:** Shadows naturally curve and deform across fluted columns, arches, and floor planes.
* **Natural Penumbra Softening:** Area emitter sampling naturally diffuses the shadow with increasing blocker distance $\Delta Z$.
