# Numerical transport corrections — 2026-10-14

These corrections address reproducible calculation inconsistencies. They do not establish visual acceptance or a physically complete renderer.

- Perspective reflection now uses each receiver’s direction toward the camera. An off-axis mirror aimed exactly at an emitter previously returned zero; the corrected analytical witness returns a positive reflection.
- Cavity attenuation now applies to exported diffuse and specular passes as well as their sum. Combined shading equals diffuse + specular + ambient.
- The compositor removes the connected isolated specular pass from a combined rig map before diffuse modulation, then adds the controlled specular contribution once. External diffuse-only maps must set lighting_map_includes_specular to false. Chroma bleed and specular kick remain artistic controls.
- Artificial near-distance attenuation caps were removed. The point fallback at 0.3 versus 0.9 distance now gives the expected 9:1 ratio.
- The grey calibration probe shares the scene area-light integrator and uses 0.18 linear reflectance. Both probe displays receive an HDR display transform and sRGB encoding.

Eight analytical tests cover reflection geometry, component accounting, single specular routing, disabled specular, inverse-square distance behavior, grey reflectance and area-light/backface behavior. CI and the pod test runner include them.

The rig uses normalized reference gain, approximate emitter beams and specular lobes, estimated single-layer depth, and artistic cavity attenuation. It is not calibrated in watts or lux. Camera intrinsics agreement between geometry estimators and reconstruction still needs explicit calibration. Removing caps can make close lights substantially brighter; exposure remains the display control. Washed-out generation results are not proven resolved by these numerical checks. No public release approval is implied.
