# Release Validation & Verification Notes

## 1. Automated Verification Suite (`v1.0.0`)

Run the full automated unit, photometric contract, and workflow verification suite from the repository root:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

### Verified Gates (97 / 97 Passing)
- **Workflow Contract & Inventory (`tests/test_workflow_contract.py`):**
  - Verifies that `workflows/` ships strictly `StudioLight_Create.json` and `StudioLight_Relight.json`.
  - Enforces coordinate convention parity (`marigold_native` on Node 14, `invert_normals_x=False` and `invert_normals_y=False` on `StudioLightRig` Node 15).
  - Verifies first-run model download defaults (`allow_download=True` on Node 58).
  - Verifies directional photometric truth (camera-right `+75°` key light illuminates `+X` facing normals).
- **Photometric Contract (`tests/test_photometric_contract.py`):**
  - Validates Schema 1 additive and subject-replacement compositing equations in linear Rec.709 space.
  - Verifies strobe-off identity, batch independence, explicit telemetry cloning, and diagnostic contact-sheet wiring.
- **Semantic Prompt & Scene Contract (`tests/test_prompt_architecture_v2.py`, `tests/test_extended_3domain.py`):**
  - Validates clause-level semantic prompt parsing across emotional markers, physiological details, and practical light sources (`RIG_OVERRIDE`, `HYBRID_PRACTICAL`, `VERBATIM_RAW`).
- **Core Shader, Area Emitters & 3D Camera-Space Shadow Raymarching (`tests/test_studiolight.py`, `tests/test_transport_math.py`):**
  - Validates Planckian locus Kelvin conversion, 6 studio modifier apertures, Turk (1990) polygonal area sampling, P0 front-of-ray occlusion gating, Interleaved Gradient Noise (IGN) dither, and CPU/MPS numerical parity.

---

## 2. Historical CUDA & Exposure Validation Archive (2026-10-14)

- **Environment:** Verified on a 64 GB CUDA pod with ComfyUI 0.37.0, frontend 1.52.7, and PyTorch 2.13.0+cu126, as well as NVIDIA RTX 3090 (24 GB VRAM) microbenchmarks.
- **HDR Headroom & Exposure:** Shader outputs retain nonnegative float32 radiance above `1.0` prior to compositor tone mapping. Under close-range high-intensity stress testing (`3200K`, `intensity=3.0`, `distance=0.8m`), pre-display linear peaks of `10.63` were confirmed to compress cleanly without hard channel clipping when exposure and filmic white point are calibrated.
