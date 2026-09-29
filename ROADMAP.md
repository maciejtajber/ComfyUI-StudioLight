# ComfyUI-StudioLight — Internal 7-Day Sprint Roadmap & Battle Plan

> **Confidential / Internal Engineering Playbook**  
> **Target Release:** Complete 3D Studio v1.0 in 7 Days  
> **Core Architecture:** Leverage proven, battle-tested open-source libraries (Zero wheels reinvented).  
> **Philosophy:** *"Pose the geometry. Place the lights. Generate the image."* (The 3D Scene is the Single Source of Truth).

---

## 1. Technical Strategy: Assemble Proven Components

To ship a complete, high-performance 3D studio in 7 days without falling into month-long engineering rabbit holes:

1. **3D Viewport Engine:** Off-the-shelf Three.js WebGL canvas loaded dynamically into a ComfyUI floating modal. Zero bundle drag on initial ComfyUI launch.
2. **Normal & Depth Baking:** Native Three.js `WebGLRenderTarget` with `MeshNormalMaterial` and linear depth shader. Bakes 1024×1024 camera-space maps in under 10ms directly to node tensors via offscreen canvas.
3. **Posing Engine (No IK Trap):** Forward Kinematics (FK) rotation using standard Three.js `TransformControls` on a standardized CC0 VRM humanoid rig, combined with 8 pre-built editorial fashion poses. No custom IK solvers to debug.
4. **3D Asset Import:** Standard Three.js `GLTFLoader` with drag-and-drop support for `.glb` / `.gltf` / `.obj`. Instant lighting support for products, cars, perfume bottles, and architectural forms.
5. **Match Pose From Image:** Lightweight 2D/3D joint estimation (MediaPipe / DWPose keypoints) mapped directly onto the 15 standard humanoid bone rotations.

---

## 2. 7-Day Sprint Schedule

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                            7-DAY EXECUTION SPRINT                            │
├─────────┬────────────────────────────────────────────────────────────────────┤
│ DAY 1   │ Latent Engine Stress-Testing:                                      │
│ (Today) │ • Test extreme lighting contrast (Chiaroscuro, Dual Rim Kickers)   │
│         │ • Dial in SI Shadows & Highlights (-0.30 shadow, +0.40 highlight)  │
│         │ • Validate linear division photometric carrier extraction          │
├─────────┼────────────────────────────────────────────────────────────────────┤
│ DAY 2-3 │ WebGL 3D Studio Modal (Three.js Stage):                            │
│ (Wed-Thu)│ • Floating ComfyUI modal triggered by [ 🎭 Open 3D Studio ]        │
│         │ • OrbitControls camera + dark studio floor grid                    │
│         │ • Interactive 3D Key, Fill, and Rim light gizmos synced to node    │
│         │ • WebGLRenderTarget offscreen baker for Camera Normal + Depth maps │
├─────────┼────────────────────────────────────────────────────────────────────┤
│ DAY 4   │ Universal 3D Asset Import:                                         │
│ (Fri)   │ • Drag-and-drop GLTF/GLB loader into viewport                      │
│         │ • Camera framing presets (Extreme Close-Up, American Plan, Full)   │
│         │ • Product & object lighting validation (perfume, car, statue)      │
├─────────┼────────────────────────────────────────────────────────────────────┤
│ DAY 5   │ Mannequin Posing & Match Pose From Image:                          │
│ (Sat)   │ • Standard VRM low-poly mannequin with FK bone rotation gimbals    │
│         │ • 8 one-click fashion & editorial pose presets                     │
│         │ • "Match Pose From Image" button: Image keypoints -> 3D mannequin  │
├─────────┼────────────────────────────────────────────────────────────────────┤
│ DAY 6   │ End-to-End Workflow Integration & Polish:                          │
│ (Sun)   │ • Pack single master workflow (StudioLight_Master_Suite.json)       │
│         │ • Test full loop: 3D Pose -> 3D Light -> Auto-Bake -> Flux Render  │
│         │ • Stress-test edge cases & UI responsiveness                       │
├─────────┼────────────────────────────────────────────────────────────────────┤
│ DAY 7   │ Packaging, Showcase Video & Public Release:                        │
│ (Mon)   │ • Record 15-second side-by-side demo (3D Light Move -> Flux Render)│
│         │ • Publish to ComfyUI Registry, GitHub, and Reddit (r/ComfyUI)      │
└─────────┴────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 0.5 Reality Check: Empirical Benchmark Summary

Validated on SimplePod (NVIDIA RTX 4090 24GB) using Flux Dev / Krea:

| Setup Tested | Pose Lock | Light Direction Match | Texture Quality | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Pure Mannequin Latent (`denoise 0.85`)** | **100%** | **High** | **Photorealistic 8K Skin** | 🏆 **Core Standard** |
| **Pure Mannequin (`0.90`) + Low Redux (`0.15`)** | **100%** | **Highest** | **Photorealistic 8K Skin** | 🥇 **Precision Stack** |
| **Contrast-Tuned Mannequin + Initial Latent** | **100%** | **High** | **Punchy Speculars** | 🌟 **Winning Pipeline** |
| **SI Shadows & Highlights Pre-Processing** | **100%** | **High** | **Controlled Shadows** | 🟢 **Shipped in v0.1** |
| Gaussian Blurred Mannequin (`0.65 - 0.75`) | 80% | Moderate | Blurry Halo Artifacts | ❌ Disproven |
| Redux + Empty Latent (`1.0`) | 30% | Weak | High Realism | ❌ Disproven (Pose drift) |
| Blurred Redux Input | 60% | Moderate | Forced Baldness / Buzzcut | ❌ Disproven (SigCLIP leak)|
| ControlNet Union (Lighting Mode) | 0% | None | Prompt-Driven Only | ❌ Disproven (CN is blind) |

---

## 4. Architectural Boundaries (What We Will NOT Build)

* ❌ **No Custom IK Solvers:** IK in browser WebGL is unstable. FK gizmos + preset poses are deterministic and robust.
* ❌ **No Blender Dependency in Production:** v1.0 bakes normals and depth directly inside the Three.js canvas in milliseconds.
* ❌ **No Mesh Auto-Rigging:** Custom 3D uploads are for static objects, products, and vehicles. Characters use the standardized VRM rig.
