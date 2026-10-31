<div align="center">

<img src="assets/studiolight_banner.png" alt="StudioLight by METAMATERIAL" width="100%" />

<br>
<br>

# StudioLight — Design the Light.

**Camera-space key, fill, and rim lights, photographic modifiers, and Kelvin control for existing photographs, finished renders, and generative drafts in ComfyUI.**

<p align="center">
  <img src="https://img.shields.io/badge/release-Consistent_V1_(R1)-1f2328?style=flat&labelColor=555555" alt="Release Consistent V1 (R1)" />
  <img src="https://img.shields.io/badge/ComfyUI-Custom_Node-1f2328?style=flat&labelColor=555555" alt="ComfyUI Custom Node" />
  <img src="https://img.shields.io/badge/FLUX.1-Dev_%C2%B7_Krea_(Optional)-1f2328?style=flat&labelColor=555555" alt="FLUX.1 Dev / Krea (Optional)" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B_CUDA_%2F_MPS-1f2328?style=flat&logo=pytorch&logoColor=white&labelColor=555555" alt="PyTorch 2.0+ CUDA / MPS" />
  <img src="https://img.shields.io/badge/python-3.10%2B-1f2328?style=flat&logo=python&logoColor=white&labelColor=555555" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/tests-68_%2F_68_passing-2ea44f?style=flat&labelColor=555555" alt="Tests 68/68 Passing" />
  <img src="https://img.shields.io/badge/license-MIT-99080a?style=flat&labelColor=555555" alt="License MIT" />
</p>

<br>

<sub>
<a href="#1-the-hook--direct-proof">1. PROOF</a> &nbsp;·&nbsp;
<a href="#2-one-image-multiple-lighting-decisions">2. SHOWCASE</a> &nbsp;·&nbsp;
<a href="#3-the-photographers-console-physical-c-stands">3. RIG CONSOLE</a> &nbsp;·&nbsp;
<a href="#4-analyze-once-adjust-preview-the-2-second-loop">4. 2-SEC LOOP</a> &nbsp;·&nbsp;
<a href="#5-add-to-the-scene--or-relight-the-subject">5. DUAL CONTRACT</a> &nbsp;·&nbsp;
<a href="#6-quick-start--workflows">6. QUICK START</a> &nbsp;·&nbsp;
<a href="#7-the-engineering-behind-the-image">7. ENGINEERING</a> &nbsp;·&nbsp;
<a href="#8-status-honest-limitations--roadmap">8. ROADMAP</a>
</sub>

</div>

<br>

---

## 1. The Hook & Direct Proof

Stop prompt gambling. StudioLight brings real, three-dimensional studio lighting control directly into ComfyUI. 

Instead of typing descriptive text and hoping the diffusion model understands your intent, you position physical lights on C-stands in 3D camera space, choose real photographic modifiers, and shape the shot with calibrated blackbody color temperatures.

https://github.com/user-attachments/assets/f82d96d9-08ac-4dfb-8f1a-a65432f33a32

<br>

> **Immediate Source → Relit Verification:** Operates on fixed 2D photographs, completed 3D renders, or newly generated drafts. The lighting proxy computes in approximately two seconds without requiring diffusion resynthesis.

---

## 2. One Image, Multiple Lighting Decisions

The core creative problem in generative AI is that changing a lighting prompt mutates the entire scene—facial features shift, clothes change cut, and geometry warps. 

StudioLight locks the image and relights the **exact same source plate** with physical precision:

<table>
  <thead>
    <tr>
      <th width="33.33%" align="center">High-Fashion Beauty</th>
      <th width="33.33%" align="center">Direct On-Camera Flash</th>
      <th width="33.33%" align="center">Golden Hour Window</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><img src="assets/showcase_high_fashion_beauty.png" width="100%" alt="High-Fashion Beauty" /></td>
      <td align="center"><img src="assets/showcase_direct_flash.png" width="100%" alt="Direct On-Camera Flash" /></td>
      <td align="center"><img src="assets/showcase_golden_hour.png" width="100%" alt="Golden Hour Window" /></td>
    </tr>
    <tr>
      <td align="center"><sub>Key Beauty Dish (5500K) + Soft Fill. Sculpted cheekbones, circular catchlights, balanced wrap.</sub></td>
      <td align="center"><sub>Hard On-Camera Flash (6500K). Micro-pinpoint glare, sharp contact drop shadows, high contrast.</sub></td>
      <td align="center"><sub>Warm Area Light (3200K) + Cool Sky Fill. Atmospheric side wrap, sunset chiaroscuro glow.</sub></td>
    </tr>
  </tbody>
</table>

### Prompt Lottery vs. Spatial Control

<table>
  <thead>
    <tr>
      <th width="50%" align="left">Prompt-Based Lighting (Text Only)</th>
      <th width="50%" align="left">StudioLight Spatial Control (VFX Engine)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><b>Subject Mutation:</b> Prompting for light changes anatomy, facial features, styling, and background geometry.</td>
      <td align="left"><b>Pure Lighting Isolation:</b> Relights fixed 2D photographs or frozen drafts without altering subject geometry or identity.</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Chromatic Bleed:</b> Color words tint skin, hair, and clothing indiscriminately across the whole image.</td>
      <td align="left"><b>Directional Lobe Isolation:</b> Key, fill, and rim colors remain strictly confined to illuminated surfaces (\(\mathbf{N} \cdot \mathbf{L} > 0\)).</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Uncalibrated Falloff:</b> Lights ignore scene depth and distance, treating all surfaces as flat planes.</td>
      <td align="left"><b>Geometric Inverse-Square Falloff:</b> Models true \((r_{\text{ref}} / r)^2\) radiometric distance attenuation calibrated to studio C-stands.</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Diffusion Bottleneck:</b> Requires 30–50 full diffusion steps just to see if a light moved by 10 degrees.</td>
      <td align="left"><b>Instant ~2s Feedback Loop:</b> Shader computes an inspectable physical lighting proxy in ~2 seconds on an RTX 3090.</td>
    </tr>
  </tbody>
</table>

---

## 3. The Photographer’s Console (Physical C-Stands)

StudioLight equips you with a professional photographer's gaffer console. You control three independent physical lights—**Key, Fill, and Rim**—in assumed 3D camera space:

* **3D C-Stand Coordinates:** Set true spherical coordinates—**Azimuth** (\(-180^\circ\) to \(+180^\circ\)), **Elevation** (\(-90^\circ\) to \(+90^\circ\)), and **Distance** in meters.
* **3-Zone 3D Aim Pointer:** Direct lights at the subject's **Upper** (face), **Center** (torso), or **Lower** zones with automatic curvature-guided aiming.
* **Blackbody Kelvin Control:** Seamlessly dial color temperatures from warm candlelight (**2000K**) to overcast daylight (**10000K**) using photographic Planckian locus chromaticity.
* **Live VFX Probes & Studio Radar:** Real-time 3D dual spheres (**18% Neutral Grey + Mirror Chrome**) and a top-down studio radar give instant visual telemetry of light vectors, fixture angles, and catchlight shapes.

<div align="center">
  <img src="assets/studiolight_probe_and_render.webp" alt="StudioLight VFX Probes & Studio Radar HUD" width="100%" />
  <br>
  <sub><b>Interactive Camera-Space Lighting Rig:</b> Real-time 3D VFX spheres &amp; top-down studio radar driving generative relighting across 6 distinct setups.</sub>
</div>

### Physically Motivated Modifier Apertures

Each modifier is modeled as a 3D virtual emitter with oriented tangent-space frames, uniform aperture sampling ($W, H, R$), cosine-weighted emission, and analytical catchlight reflections:

<div align="center">
  <img src="assets/01_emitter_architecture.svg" alt="StudioLight Geometric Area Emitter Architecture" width="735" />
</div>

<table>
  <thead>
    <tr>
      <th width="18%" align="left">Modifier</th>
      <th width="20%" align="center">Aperture<br>Dimensions</th>
      <th width="14%" align="center">Beam&nbsp;Angle<br>(&theta;<sub>beam</sub>)</th>
      <th width="26%" align="left">Optical Approximation</th>
      <th width="22%" align="left">Catchlight Profile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left"><b>Direct On-Camera Flash</b></td>
      <td align="center">0.08m&nbsp;×&nbsp;0.05m</td>
      <td align="center">70°</td>
      <td align="left">Near-point strobe emitter</td>
      <td align="left">Crisp micro-pinpoint glare</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Hard Snoot</b></td>
      <td align="center">R&nbsp;=&nbsp;0.05m<br>(10cm)</td>
      <td align="center">18°</td>
      <td align="left">Collimated beam cone</td>
      <td align="left">Tight circular isolation</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Fresnel Spotlight</b></td>
      <td align="center">R&nbsp;=&nbsp;0.15m<br>(30cm)</td>
      <td align="center">35°</td>
      <td align="left">Optical lens edge cutoff</td>
      <td align="left">Compact stepped lens disc</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Stripbox</b></td>
      <td align="center">0.30m&nbsp;×&nbsp;1.20m</td>
      <td align="center">65°</td>
      <td align="left">1:4 Asymmetric rectangular emitter</td>
      <td align="left">Elongated vertical catchlight bar</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Beauty Dish (Silver)</b></td>
      <td align="center">R&nbsp;=&nbsp;0.275m<br>(55cm)</td>
      <td align="center">85°</td>
      <td align="left">Parabolic dish + center deflector (36% radius ratio, 12% transmission)</td>
      <td align="left">Sculpted facial wrap, circular donut ring</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"><b>Octabox (Soft)</b></td>
      <td align="center">R&nbsp;=&nbsp;0.70m<br>(140cm)</td>
      <td align="center">130°</td>
      <td align="left">Broad Lambertian octagonal emitter</td>
      <td align="left">Large wrap disc, soft continuous penumbra</td>
    </tr>
  </tbody>
</table>

---

## 4. Analyze Once. Adjust. Preview. (The 2-Second Loop)

The central workflow breakthrough of StudioLight is separating scene geometry analysis from lighting adjustments:

1. **Analyze Once:** When you load an image, StudioLight extracts its surface orientation, relative depth, and intrinsic albedo.
2. **Adjust & Preview in ~2 Seconds:** With geometry cached, you move C-stands, swap modifiers, and tweak Kelvin temperatures. The shader computes the complete 3D relit proxy in **approximately 2 seconds** on an NVIDIA RTX 3090.
3. **Zero Diffusion Bottleneck:** You explore multiple lighting setups in seconds without running heavy KSampler passes. When you find the perfect light, you optionally pass the proxy to FLUX for final material synthesis.

### RTX 3090 Performance Benchmarks (Shader Parameter Study)

<table>
  <thead>
    <tr>
      <th align="left">Resolution</th>
      <th align="left">Configuration Tier</th>
      <th align="center">Samples&nbsp;(S)</th>
      <th align="center">Steps</th>
      <th align="center">Pass&nbsp;Latency (RTX&nbsp;3090)</th>
      <th align="center">Peak&nbsp;VRAM</th>
      <th align="center">Ray-Sample Steps&nbsp;/&nbsp;sec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">512&nbsp;×&nbsp;512</td>
      <td align="left">Fast Preview</td>
      <td align="center">4</td>
      <td align="center">12</td>
      <td align="center">12.0 ms</td>
      <td align="center">113 MB</td>
      <td align="center">1,047 M/s</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"></td>
      <td align="left">Balanced (Default)</td>
      <td align="center">8</td>
      <td align="center">16</td>
      <td align="center">29.6 ms</td>
      <td align="center">118 MB</td>
      <td align="center">1,135 M/s</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left">1024&nbsp;×&nbsp;1024</td>
      <td align="left">Fast Preview</td>
      <td align="center">4</td>
      <td align="center">12</td>
      <td align="center">32.0 ms</td>
      <td align="center">450 MB</td>
      <td align="center">1,571 M/s</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"></td>
      <td align="left">Balanced (Default)</td>
      <td align="center">8</td>
      <td align="center">16</td>
      <td align="center">83.0 ms</td>
      <td align="center">469 MB</td>
      <td align="center">1,617 M/s</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left"></td>
      <td align="left">High-End Quality</td>
      <td align="center">16</td>
      <td align="center">24</td>
      <td align="center">245.3 ms</td>
      <td align="center">469 MB</td>
      <td align="center">1,642 M/s</td>
    </tr>
    <tr></tr>
    <tr>
      <td align="left">2048&nbsp;×&nbsp;2048 (4.2&nbsp;MP)</td>
      <td align="left">Balanced (Default)</td>
      <td align="center">8</td>
      <td align="center">16</td>
      <td align="center">311.5 ms</td>
      <td align="center">1.87 GB</td>
      <td align="center">1,724 M/s</td>
    </tr>
  </tbody>
</table>

---

## 5. Add to the Scene — Or Relight the Subject

StudioLight features an explicit **Schema 1 Compositing Contract**, allowing you to choose how the new light integrates with the existing image:

### Operation A: Add Studio Light (Default)
$$\text{HDR} = \text{Source} \cdot 2^{\text{exposure}} + \text{Diffuse} + \text{Specular} + \text{Ambient}$$
* **Preserves Native Environment:** The original scene is retained as an independent underlying base layer. Existing practical lamps, candles, neon signs, daylight skies, and ambient reflections survive natively.
* **Zero False Emission Guessing:** Does not guess emitters from bright highlights, preventing skin highlights or light spill from turning into artificial light sources.

### Operation B: Replace Subject Light (Explicit Mode)
$$S_{\text{retained}} = \text{Source} \cdot 2^{\text{exposure}} \cdot \left((1 - M) + p \cdot M\right)$$
* **Targeted Portrait Relighting:** Isolates the subject with a high-precision mask ($M$) and applies new studio illumination over pure albedo.
* **Controlled Source Preservation ($p$):** Dial the `source_preservation` slider ($0.0$ to $1.0$) to choose how much original subject texture and existing reflected illumination to retain.

---

## 6. Quick Start & Workflows

### 1. Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maciejtajber/ComfyUI-StudioLight.git
pip install -r ComfyUI-StudioLight/requirements.txt
```
*(PyTorch `torch>=2.0.0` is treated as a host ComfyUI dependency so your local CUDA / MPS environment is preserved).*

### 2. Analysis Dependencies & Models
* **Normals & Depth:** Install `comfyui_controlnet_aux` for `DSINE` / Marigold Normals and `DepthAnythingV2Preprocessor`.
* **Intrinsic Albedo:** Install `ComfyUI-StudioLight-DeLight` (model: `prs-eth/marigold-iid-lighting-v1-1`).
* **Subject Masking:** Install `comfyui-rmbg` (`BiRefNet-general`).
* **Optional Diffusion Refinement:** `flux1-dev.safetensors` or `flux1-krea-dev.safetensors`.

### 3. Verification Suite
```bash
python3 -m unittest discover -s tests
```
*(68 / 68 automated unit & regression tests passing).*

### 4. Ready-to-Run Workflows
* **StudioLight Relight (`workflows/StudioLight_PhotoRelight.json`):** Single-pass workflow to load any photograph or render and relight it in ~2 seconds.
* **StudioLight Studio (`workflows/StudioLight_Master_Studio.json`):** Flagship pipeline featuring prompt decoupling, 3D C-stands, and optional FLUX Krea Dev generative refinement.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              STUDIOLIGHT WORKFLOW TOPOLOGY                             │
├────────────────────────┬───────────────────────────┬───────────────────────────────────┤
│ 1. INGESTION & DEPTH   │ 2. PHYSICAL 3D RIG        │ 3. CONSISTENT COMPOSITOR & OUTPUT │
├────────────────────────┼───────────────────────────┼───────────────────────────────────┤
│ • Fixed Photo / Draft  │ • Key / Fill / Rim Stands │ • Direct Relit Proxy (~2s)        │
│ • Marigold Normals     │ • Octa / Dish / Flash     │ • Pure Additive Mode (Schema 1)   │
│ • Depth Anything V2    │ • Kelvin Blackbody (CCT)  │ • Controlled Subject Replacement  │
│ • Marigold IID Albedo  │ • 3D Shadow Raymarching   │ • (Optional) FLUX Refinement      │
│ • BiRefNet Mask        │ • Full-Surface Telemetry  │   (Denoise 0.35–0.45)             │
└────────────────────────┴───────────────────────────┴───────────────────────────────────┘
```

---

## 7. The Engineering Behind the Image

StudioLight represents an extensive engineering effort to bring physically motivated VFX techniques into diffusion pipelines.

### Dual-Input Architecture

<div align="center">
  <img src="assets/02_studiolight_relighting_pipeline.svg" alt="StudioLight Dual-Input Pipeline" width="700" />
</div>

### 6-Stage Technical Layer Dissection

<div align="center">
  <img src="assets/studiolight_pipeline_dissection_grid.png" alt="StudioLight 6-Stage Pipeline Dissection" width="100%" />
  <br>
  <sub><b>Six-Stage Technical Pipeline Dissection:</b> Row 1: Source Plate ➔ Estimated Depth Map ➔ Surface Normals. Row 2: Pure Relighting Field ➔ Lit Proxy Plate (~2s) ➔ Final FLUX Synthesis.</sub>
</div>

### 3D Camera-Space Shadow Raymarching
* **P0 Front-of-Ray Occlusion Gate ($D_{\text{scene}} \le D_{\text{ray}} - \epsilon$):** Evaluates depth strictly in front of the ray sample, paired with a $0.45\text{m}$ finite-thickness boundary and bilateral depth-guided filtering to eliminate false behind-ray occlusions and self-shadow acne.
* **Interleaved Gradient Noise (IGN) Dither:** Smooths discrete step transitions across expanding depth gaps without artificial blur passes.

<div align="center">
  <img src="assets/stress_02_slit_banding_vs_dither.png" width="95%" alt="3D Camera-Space Shadow Raymarching: Banding vs IGN Dither" />
  <br>
  <sub>Slit Aperture Penumbra Stress Test (Row 1: S=4 no dither, S=8 no dither, S=4 dither. Row 2: S=8 dither [Balanced Default], S=16 dither [High-End], S=64 Reference).</sub>
</div>

### Core Engine Components (Clean Names)

* **StudioLight Rig:** 3-point C-stand console calculating area diffuse, specular microfacets, and 3D shadow raymarching.
* **Consistent Compositor:** Physical layer engine executing Schema 1 Additive and Replacement modes, directly driving the ~2s preview.
* **DeLight Controller:** Blends source appearance with estimated Marigold albedo to eliminate harsh baked-shadow doubling.
* **Subject Mask Isolator:** Automatic BiRefNet foreground segmentation for clean subject isolation.
* **Prompt Sanitizer:** Strips conflicting lighting tokens to eliminate prompt clashing during optional Pass 2 diffusion.

### Five Working Rules for Production
1. **Choose Your Relighting Mode First:** Use **Additive Mode** to add key/rim strobes while keeping natural environmental lighting. Use **Replacement Mode** with `source_preservation` when you need to completely reshape key lighting on a portrait.
2. **Keep Source Plate & Diffuse Base Independent:** The source carrier preserves environmental background radiance; the DeLight Controller supplies pure diffuse reflectance.
3. **Inspect Upstream Geometry Before Tuning Intensities:** Verify surface normals and relative depth maps before over-amplifying rig power.
4. **Trust the Direct Proxy:** The direct ~2s compositor output is physically coherent. If light vectors or shadows look misplaced in the proxy, generative diffusion will not correct them.
5. **Diffusion is an Optional Material Synthesizer:** Use Pass 2 FLUX Krea Dev at gentle denoise (`0.35 – 0.45`) to add dielectric micro-textures and pore-level skin scattering.

---

## 8. Status, Honest Limitations & Roadmap

### Honest Engineering Boundaries
* **Single-View 2.5D Frustum:** Geometry and shadow raymarching operate on surfaces visible to the camera. Surfaces completely occluded in the depth map cannot cast shadows.
* **Estimated Intrinsic Albedo:** Single-image albedo decomposition via Marigold IID is an estimate. Extreme specular glare or deep baked shadows may leave subtle residual tones in challenging plates.
* **Additive Layer Retention:** Additive mode retains existing ambient lighting and practical lamps natively. It does not erase old light sources or rotate existing baked shadows.
* **Diffusion Interpretation:** Higher Pass 2 denoise levels ($\ge 0.60$) give diffusion greater freedom to repaint textures, whereas lower values ($0.35–0.45$) strictly preserve subject identity and proxy lighting vectors.

### Status-Based Capability Roadmap

| Capability / Milestone | Engineering Status | Architectural Scope |
| :--- | :--- | :--- |
| **Fixed-Image 2D Relighting** | **Shipped in Consistent V1** | Relight arbitrary photographs and renders without mandatory diffusion passes. |
| **Pure Additive Layering Contract** | **Shipped in Consistent V1** | Native practical and room-ambient preservation ($HDR = S \cdot 2^e + D + P + A$). |
| **Controlled Subject Replacement** | **Shipped in Consistent V1** | Explicit `source_preservation` slider with BiRefNet subject isolation. |
| **Intrinsic Albedo Pipeline** | **Shipped in Consistent V1** | Full-surface Marigold IID De-Light modulation via the DeLight Controller. |
| **Direct Proxy / VAE Parity** | **Shipped in Consistent V1** | Preview and VAEEncode receive identical, unclipped linear HDR data in ~2s. |
| **Optional FLUX Krea Polish** | **Shipped in Consistent V1** | Decoupled Pass 2 generative refinement stage for skin micro-textures. |
| **Temporal Light Keyframing** | **Next Sprint (Q4 2026)** | Reproducible interpolation of C-stand coordinates and Kelvin across video frame sequences. |
| **Video Diffusion Integration** | **Next Sprint (Q4 2026)** | Injecting 32-bit linear photometric plates into spatio-temporal video attention streams (Wan / LTX). |
| **Temporal Geometry Deflicker** | **Next Sprint (Q4 2026)** | Depth and normal temporal filtering to eliminate multi-frame edge chatter. |
| **Multi-View Coherent Rigging** | **Research Horizon** | Shared 3D camera-space lighting rigs across multi-camera array captures. |
| **Integrated Three.js 3D Mesh Stage** | **Explicitly Retired** | Single-view depth raymarching supersedes the legacy Three.js mannequin stage. |

---

## License & Credits

* **Code & Shaders:** MIT License. Created by **Maciej Tajber** // **METAMATERIAL** Applied VFX Systems.
* **Ecosystem Credits:** Built on the open research of the ComfyUI, Depth Anything V2, Marigold, BiRefNet, and FLUX communities.
