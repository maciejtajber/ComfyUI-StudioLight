# Photometric compositor contract

The compositor works with linear RGB lighting **gains** and linearized sRGB plates. A generated/processed sRGB plate is not a calibrated scene-radiance or albedo measurement. Relative depth supplies an assumed-scale camera-space proxy, not measured room dimensions.

## Data path and compatibility

`StudioLightRig` preserves outputs 0–3 and adds `telemetry` at output 4. ComfyUI calls `execute_rig_with_telemetry`; existing Python `execute_rig(...)` calls still return four values. The compositor preserves `lit_image` at output 0 and adds `diagnostic_contact_sheet` at output 1. Python callers should use indexed outputs or unpack both compositor results.

Connect rig `telemetry` directly to compositor `telemetry`. The supplied Master and Option A workflows include this connection, the depth fallback and the diagnostic preview. Existing image-only workflows still use the legacy tensor attribute when available. A clone, saved PNG or other image transformation may discard that attribute; the explicit socket survives those image operations. When intentionally changing the light field, update its corresponding telemetry too rather than passing stale separated fields.

Telemetry schema 2 contains:

- `direct_diffuse`, `direct_specular`, `ambient_rgb`: nonnegative BHWC linear gains. The first two exclude ambient. `direct_specular` uses the same raw scale as rig output 3.
- Optional `d_norm`, `fg_confidence`, `shadow_occlusion`: BHW1 fields.
- Optional `volumetric_beam`: BHWC RGB field.
- `d_split`: scalar for one frame, B×1×1×1 for batches.
- `depth_convention = relative_near_one`, `geometry_scale = assumed`.

The compositor processes each batch frame independently. Inputs must have matching batch sizes or singleton batches; all singleton fields expand consistently. Invalid shapes, nonfinite tensors, incompatible batch sizes and out-of-range scalar controls raise useful `ValueError`s. IMAGE input is RGB sRGB in [0,1]; lighting is nonnegative HDR RGB. Recovery MASK is BHW or BHW1.

## Strength and specular semantics

With schema-2 telemetry, `lighting_strength` scales direct strobe light, specular and haze while the rig's ambient term is independent. At zero strength, non-emissive foreground has only its declared ambient/floor. Existing background exposure and protected practical emission remain. A whole-image bright-channel/local-contrast guard protects coloured practicals even when depth assigns them to the background; `emission_mask` overrides that heuristic (an all-zero mask disables it). Bright reflectors can resemble emitters, so this is not semantic emission detection. Cast-shadow and cyclorama adjustments fade to zero at zero strength. This is a foreground replacement/background preservation compositor, not an additive flash exposure of the entire original plate.

A matching `specular_map` is removed from a legacy combined map before diffuse multiplication when `lighting_map_includes_specular=True`. Set that flag false for a diffuse-only input. Schema-2 telemetry already supplies the separated diffuse component, avoiding subtraction uncertainty. If no explicit specular map is connected, telemetry supplies its raw specular pass. With neither, the legacy threshold highlight approximation remains.

`SPEC_PASS_REFERENCE=0.25` is the historical artistic catchlight normalization; it is not a unit conversion and is not used when subtracting raw specular from a combined map.

## Optional room fill

`room_bounce_strength` defaults to 0 so old workflows do not silently acquire assumed room fill. A useful starting experiment is 0.22, but inspect the diagnostic panel and compare against 0.

Bounce is an additive, per-image RGB fill derived only from direct diffuse strobe gain on visible non-sky background candidates. It scales with strength, wall_reflectance, relative-depth proximity and sample confidence. It never reduces direct illumination. Zero strobe, black reflectance, missing strobe telemetry, constant depth or insufficient subject/background evidence produces zero bounce.

`enclosure_override=-1` uses the relative-depth heuristic; 0 marks open space, and 1 requests enclosed-space fill. Visible candidate surfaces and usable source light are still required. `wall_reflectance` is a neutral scalar assumption, not material recovery. RGB colour follows incident strobe light; this implementation does not estimate wall pigment from a photograph or model invisible walls. The estimate is **not metric global illumination or an energy-conservation proof**.

## Material and colour controls

No automatic warm-skin injection or blanket face-shadow lifting is applied. Existing crevice de-lighting, glare neutralization, dark-material safeguards, haze and cyclorama guards remain.

`shadow_recovery` defaults to 0 and requires `recovery_mask`. Select exactly one material, including some reliably exposed pixels. The prior uses the masked material's median colour, is capped at 0.05 linear reference luminance and affects only dark masked pixels. It is independent of the new strobe strength. With insufficient exposed samples it does nothing. It cannot recover detail that was clipped to black.

`black_gate` defaults to Legacy material protection. Low-signal chroma preservation uses the proposed lower linear thresholds as an explicitly selectable photographic tuning, not a claimed gamma conversion or universal material classifier.

`tone_mapping` preserves Legacy per-channel as its default. Luminance HDR maps luminance and compresses out-of-gamut chroma toward neutral while preserving mapped luminance; RGB ratios remain unchanged only while already inside gamut. `exposure_ev` applies to composed linear HDR before tone mapping. Nonlinear highlight normalization means changing illumination and changing exposure are not equivalent with artistic specular enabled.

Legacy Soft Light, background_protection and exposure arguments are supported again. Soft Light is an artistic display-space overlay: gain=1 is neutral at lighting_strength=1, blend_amount=0 returns the input. Use the separate Post-Composite node for its full retouching controls.

## Diagnostic contact sheet

Top row: original sRGB draft; fixed-exposure luminance-tone-mapped lighting field.
Bottom row: lit proxy; false-colour diagnostics.

- Red: applied shadow occlusion, after strength/haze modulation.
- Green: actual bounce luminance mapped by `b/(1+b)`.
- Blue: candidate-sample confidence. This is not proof that the scene is enclosed.

Each tile has a 20-pixel label bar. Tile size is max(1,H//2) by max(1,W//2), so odd and 1-pixel images are supported. The sheet is a display visualization, not a quantitative HDR export. Preserve raw floats for numerical comparisons.

## Validation

Run `python test_studiolight.py`, then:

```sh
python -m unittest test_workflow_migration test_release_candidate test_pod_harness test_visual_corrections test_transport_math test_softlight test_photometric_contract -v
```

The new contract tests cover additive/coloured/power-linear fill, black reflectors, missing geometry, strobe-off foreground/background/cyclorama behavior, independent ambient, raw specular accounting, reconstruction restrictions, batch equivalence, metadata survival, tiny/odd previews, finite validation, unchanged inputs and graph wiring. CUDA remains a separate release gate through the pod harness; CPU success is not GPU or diffusion visual sign-off.

Several old assertions needed corrected fixtures: exposure linearity disables nonlinear artistic specular; neutral gain explicitly sets strength=1; the in-gamut colour test uses a patch that actually fits the selected curve; Kelvin tests isolate diffuse colour; the specular test explicitly strips metadata for its legacy comparison and checks telemetry auto-selection against the raw pass. No failure was skipped to make the suite green.
