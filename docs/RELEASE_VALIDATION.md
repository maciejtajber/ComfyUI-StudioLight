# RTX 3090 release validation: sanitizer, HDR and batches

This is an **unreleased candidate**. The original harness instructions are preserved below; completed 2026-10-14 checks are recorded in the appended result section. Maciej accepted the final exposure-confirmation result visually on 2026-10-14. Production registration stays at four nodes. Shader outputs now retain nonnegative float32 radiance above 1; display PNG previews can clip this data. PhotometricComposite remains responsible for modulation, shoulder compression and display encoding. Neither chroma_bleed nor specular_kick restores radiance lost upstream.

## Pod prerequisites

RTX 3090 (24 GB VRAM), container disk **at least 80 GB** plus room for several GB of diagnostic captures. Install the current candidate under ComfyUI/custom_nodes/ComfyUI-StudioLight. Use the pod's ComfyUI Python with CUDA PyTorch, numpy and Pillow. Install comfyui_controlnet_aux for DSINE and DepthAnythingV2. Keep the ComfyUI queue empty before testing.

The runner uses the master workflow's FLUX Krea Dev, CLIP-L, fp16 T5 and ae.safetensors. It verifies selected files against `/object_info`; missing models stop the run. Alternative Krea/T5 filenames must be explicitly selected using `--unet` and `--t5`; no model-family substitution occurs. Codex will resolve real pod paths once connection details arrive.

## Execution

Before starting/restarting ComfyUI, install the **test-only** capture/clamp helpers:

```bash
python tools/prepare_pod_release_tests.py --comfy-root /actual/path/to/ComfyUI
```

The installer refuses to replace a different existing helper. It creates a separate StudioLightReleaseTests package; production mappings are untouched. Start/restart ComfyUI, then run:

```bash
python tools/run_pod_release_tests.py --url http://127.0.0.1:8188 --out /actual/session/path/StudioLightRelease
```

Use `--plan-only` to write the manifest without contacting ComfyUI or consuming GPU time. `--resume` skips completed cases only when commit, workflow hash and settings match. Incomplete cases may be rerun. Timeouts do not cancel server jobs: inspect the recorded prompt ID/history before resuming. Changed candidates require new output folders.

## A. Automatic numerical gates

The runner first requires CUDA and runs migration regressions plus the six new release tests, using CUDA for their tensor operations. It also runs the original 22-category suite, which retains explicit CPU fixtures; this does not certify every original fixture on CUDA. Failure stops diffusion rendering.

Required results:

- Current 26–31-widget layouts preserve supplied settings, legacy 22/26 layouts migrate correctly, invalid settings/slots are rejected, and API specular slot 3 remains valid.
- 2-normal/3-depth, 3-normal/2-depth and empty batches raise clear errors.
- 1:2, 2:1 and 2:2 geometry inputs produce exactly two outputs matching solo output within 1e-5.
- Compositor rejects incompatible batches and supports singleton broadcasting.
- Depth-connected and depthless lighting remains finite/nonnegative; raw energy above 1 survives, matching diffuse + specular + ambient when AO is disabled.
- Distinct bright fields produce distinct composite values; old clamping makes them identical.
- Actual master graph sanitizer outputs feed the correct draft/final prompt inputs; serialized links agree; draft removes gloss/patent words and final text preserves the master prompt.

The server itself must report CUDA. Tests record numerical logs, server stats, and the candidate commit/workflow hash.

## B. 24-render factorial

Two seeds **777777 / 424242** × three presets **Cinematic Film Noir / High-Fashion Beauty / Direct On-Camera Flash** × sanitizer off/on × legacy clamp/HDR preservation = **24 finals**.

| Arm | Draft sanitizer | Rig-to-compositor lighting/specular |
|---|---|---|
| S0 H0 | Bypassed, previous prompt + suffix | Clipped [0,1] baseline |
| S0 H1 | Bypassed | HDR preserved |
| S1 H0 | Connected | Clipped [0,1] baseline |
| S1 H1 | Connected | HDR preserved: candidate |

Baseline clamping is reproduced by isolated test nodes, not by mutating production code. Both sampler seeds use the case seed. All other master settings stay fixed: 1024², 30 steps, CFG 1, Euler/simple, guidance 3.5, split step 10, Pass 2 denoise .55, lighting strength 1.2, chroma_bleed .45, specular_kick .45, white point 3.5 and unchanged darkroom/geometry settings.

Compare H0/H1 within the same seed/preset/sanitizer to isolate clipping. Compare S0/S1 within a fixed HDR state to isolate draft cleanup. Sanitizer changes intentionally alter draft generation; assess identity and materials, not just brightness.

## C. Four close-light stress renders

Both seeds with sanitizer on, old clamp vs HDR. Custom key-only Beauty Dish (Silver), intensity **3.0**, distance **0.8 m**, azimuth **0°**, elevation **20°**, **3200 K**; fill/rim disabled. Other settings unchanged. These force bright warm illumination to reveal clipping and chromaticity loss. Total: **28 finals**.

## Evidence and automatic comparison gates

Each case saves exact API prompt, prompt ID/history, completion record, final PNG, and compressed float32 `.npz` plus JSON metrics for draft, depth, normals, raw lighting, raw specular, lighting used, specular used, composite, darkroom and final image. Metrics include shape, min/max, mean, p99, fraction of channels above 1, and saturated display-pixel fraction.

The runner checks that HDR pairs share the same draft/depth/normals within 1e-6; HDR arms use raw fields, legacy arms use exactly clamped fields. It saves `comparison_gates.json`, `summary.json`, and a 4-column × 7-row contact sheet. Rows 1–6 show S0H0 / S0H1 / S1H0 / S1H1; row 7 shows two stress pairs.

## D. Visual sign-off

Execution success is not an aesthetic verdict. Inspect full-resolution finals and intermediate composites for face/hand anatomy, identity, coat construction, handbag material, forehead/cheek highlight gradients, warm/cool color retention, broad white patches, collar seams, background cast shadows and rim placement. Determine whether HDR detail survives final diffusion and whether sanitizer reduces baked highlights while retaining desired material appearance in Pass 2.

A higher saturated-pixel count is not automatically failure: legitimate catchlights may saturate. Assess it alongside raw fields and visual evidence. If HDR changes exposure too much, run a separate exposure-calibration experiment afterward; do not tune chroma_bleed/specular_kick during this factorial.

## E. Browser and publication gates

Load the UI master workflow and verify sanitizer connections. Select presets and confirm displayed settings. Adjust a lamp, confirm preset becomes Custom, queue a candidate, save/reload and confirm settings/links survive. Browser behavior requires a live browser session and is not certified by the API harness.

After numerical, visual and browser acceptance, rebuild the archive from the final tracked commit and tag/publish that revision. Back up all pod outputs, float passes, prompts, manifest, contact sheet and logs locally and to the established R2 session archive **before terminating the pod**.


## Fixed-draft visual correction replay

Install the updated test-only helpers with `prepare_pod_release_tests.py`, then restart the test server when its queue is empty. `StudioLightTestLoad` reads only finite NPZ images inside ComfyUI's output directory. Keep the original 28-case captures on the pod.

```bash
python tools/run_visual_corrections.py --baseline /path/to/original/results --out /path/to/new/correction/results --url http://127.0.0.1:8188 --source-session STUDIOLIGHT_RELEASE_20261013
```

Eight renders reuse exact saved float32 drafts, depth and normals: cases 17 and 19 with background protection; 26 and 28 at -1/-2 EV; Noir case 4 and Beauty case 8 as regression checks. Each capture checks that draft, geometry and raw lighting match the original to 1e-6. The manifest records the tested compositor SHA256. Existing completed correction cases are skipped; use a fresh output folder after any code/settings change. These controls require visual inspection; numerical passes do not constitute artistic acceptance.

Add `--matrix natural` to run four additional neutral/warm comparisons on the same two stress drafts: Octabox key at 1.8 m/intensity 1, gentle fill, no rim, normal boost 1, catchlight kick 0.15, chroma bleed 0.65 and 0 EV. This deliberately changes the light setup; replay equality is required for draft/depth/normals, not for the lighting fields. Use a separate output folder for each matrix.

### Separating warmth, refinement and downstream clipping

Additional portable matrices use fresh output directories:

- `--matrix color`: six renders comparing 3200K versus 5500K at matched raw-light luminance and final denoise 0.55 versus 0.35, on both stress seeds. The 5500K intensity is 1.950249 rather than 3 to compensate Kelvin conversion luminance, preserving reference brightness.
- `--matrix highlight`: six renders turning the post-tone-map highlight boost off, comparing warm and matched neutral at -2 EV plus the original warm setup at 0 EV.
- `--matrix safe_darkroom`: two accepted regular-preset checks (Noir/Beauty) with the highlight boost neutral.

The original +0.15 darkroom boost is retained in the stored baseline prompts; these tests change it explicitly. Compare composite, darkroom and final floats separately. With all darkroom adjustments neutral, composite and darkroom must match exactly. Warmth is an intended light-temperature response; display-channel clipping is a separate loss of information. Standard workflows now default to neutral highlight adjustment.


---

## Completed validation — 2026-10-14

Runtime source 6502f9f was hash-verified on a fresh 64 GB CMP 170HX pod with ComfyUI 0.37.0, frontend 1.52.7 and PyTorch 2.13.0+cu126. All 33 numerical regressions passed on CUDA and CPU; the original 22 verification categories passed locally. GitHub CI passed. Packaging, rig workflow migration, and browser preset selection/Custom switching were checked.

The shipped master completed successfully. An exact-input replay of the close 3200 K/intensity 3/0.8 m Beauty Dish stress case completed successfully but looked overexposed at 0 EV. Its pre-display linear peak was 10.63. Its display composite had no channel at exactly 1, while 27.87% of pixels had at least one channel >=0.99. Do not describe this near-white statistic as a hard-clipping count.

A single -3.75 EV confirmation kept the same draft, depth, normals and raw diffuse/specular lighting exactly. Composite and final pixels with a channel >=0.99 fell to zero. The final maximum was 0.93164. This exposure was computed to keep the measured maximum below the legacy shoulder threshold; it is specific to this test and is not the master default. Warm Kelvin output remains intentional.

The default master uses legacy tone mapping and background protection off. Other exposure/light combinations can deliberately produce near-white highlights. The renderer is approximate and intensity is a normalized reference gain. These checks do not guarantee physically complete simulation, all workflow combinations, or all diffusion color behavior. Maciej accepted this final result visually on 2026-10-14. Acceptance covers the demonstrated result and does not guarantee every possible light, prompt, or material.

Fresh-pod geometry setup must include timm for DSINE, in addition to the preprocessor package’s requirements. Preserve the host CUDA PyTorch build during dependency installation.

No public repository release, registry submission, or pod termination was performed by these checks.
