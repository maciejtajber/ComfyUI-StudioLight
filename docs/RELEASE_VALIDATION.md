# Release validation — 2026-10-14

Runtime source 6502f9f was hash-verified on a fresh 64 GB CMP 170HX pod with ComfyUI 0.37.0, frontend 1.52.7 and PyTorch 2.13.0+cu126. All 33 numerical regressions passed on CUDA and CPU; the original 22 verification categories passed locally. GitHub CI passed. Packaging, rig workflow migration, and browser preset selection/Custom switching were checked.

The shipped master completed successfully. An exact-input replay of the close 3200 K/intensity 3/0.8 m Beauty Dish stress case completed successfully but looked overexposed at 0 EV. Its pre-display linear peak was 10.63. Its display composite had no channel at exactly 1, while 27.87% of pixels had at least one channel >=0.99. Do not describe this near-white statistic as a hard-clipping count.

A single -3.75 EV confirmation kept the same draft, depth, normals and raw diffuse/specular lighting exactly. Composite and final pixels with a channel >=0.99 fell to zero. The final maximum was 0.93164. This exposure was computed to keep the measured maximum below the legacy shoulder threshold; it is specific to this test and is not the master default. Warm Kelvin output remains intentional.

The default master uses legacy tone mapping and background protection off. Other exposure/light combinations can deliberately produce near-white highlights. The renderer is approximate and intensity is a normalized reference gain. These checks do not guarantee physically complete simulation, all workflow combinations, or all diffusion color behavior. Final visual acceptance remains the operator’s decision.

Fresh-pod geometry setup must include timm for DSINE, in addition to the preprocessor package’s requirements. Preserve the host CUDA PyTorch build during dependency installation.

No public repository release, registry submission, or pod termination was performed by these checks.
