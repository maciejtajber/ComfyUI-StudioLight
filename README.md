# ComfyUI-StudioLight

> **Professional Photographic & Fashion 3-Point Studio Lighting Controller for ComfyUI.**  
> Stop fighting prompt RNG for studio lighting. Shape Key, Fill, and Rim lights with real photographic angles, Kelvin color temperatures, optical modifiers, and live 2D radar control.

![ComfyUI Node](https://img.shields.io/badge/ComfyUI-Custom%20Node-000000?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-black?style=flat-square)
![Optimized for: Flux.1 & SDXL](https://img.shields.io/badge/Optimized%20for-Flux.1%20%7C%20SDXL-black?style=flat-square)

---

## Why this exists

Prompting lighting with text alone (*"dramatic rim light, soft key"*) is unreliable diffusion RNG. Words bleed into anatomy, alter facial identity, and cannot guarantee where photons physically strike. Existing relighting tools either hallucinate unwanted pixels or force clunky mathematical matrix sliders.

**`StudioLight`** bridges real-world photographic studio physics into ComfyUI:
1. **Interactive 2D Studio Radar:** Visual top-down radar on the node face with click-and-drag angle adjustment.
2. **Physical Deferred Relighting Map:** 32-bit Half-Lambert wrap diffuse + Blinn-Phong specular shading over surface normals for ControlNet, IC-Light, and latent guides.
3. **High-Signal Studio Tokens:** Zero prompt pollution. Generates concise, curated photographic tokens tested for Flux.1 and SDXL attention.
4. **Step-Scheduled Conditioning:** Native integration with `ComfyUI-PromptBlock` (fires at late steps `0.40 → 1.00`, `A_NORM` magnitude normalization) so lighting shapes reflections without distorting face geometry.
5. **VFX Probe Preview Card:** Instant 0.03-second generation of an 18% Grey Matte Ball + Mirror Chrome Ball + 2D Studio Floorplan Diagram.

---

## Architecture & Wiring

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 StudioLightRig                         │
                  ├────────────────────────────────────────────────────────┤
 (IMAGE, opt) ──► │ [●] normals_optional                                   │
 (IMAGE, opt) ──► │ [●] depth_optional                                     │
 (CLIP, opt)  ──► │ [●] clip_optional                                      │
 (COND, opt)  ──► │ [●] conditioning_in (from PromptBlock)                 │
                  │                                                        │
                  │   Preset: [ Rembrandt Classic                      ▼ ] │
                  │                                                        │
                  │   [●] Interactive 2D Studio Radar (Live Canvas)        │
                  │                                                        │
                  │   KEY LIGHT:  +45° Azimuth | 35° Elev | 5500K | Octa   │
                  │   FILL LIGHT: -30° Azimuth | 15° Elev | 4800K | Octa   │
                  │   RIM LIGHT: +160° Azimuth | 45° Elev | 6500K | Strip  │
                  ├────────────────────────────────────────────────────────┤
                  │ [●] relighting_map ────► [ControlNet / IC-Light]       │
                  │ [●] probe_preview  ────► [PreviewImage] (Instant VFX)  │
                  │ [●] conditioning   ────► [FluxGuidance / KSampler]     │
                  │ [●] lighting_prompt ───► [ShowText / CLIPTextEncode]   │
                  └────────────────────────────────────────────────────────┘
```

---

## The Studio Rig Controls

### 1. Three-Point Studio Rig
* **Key Light:** Primary directional light defining form, shadows, and catchlights.
* **Fill Light:** Shadow softening and contrast control (independent intensity and Kelvin).
* **Rim / Kicker:** Rear edge separation defining silhouettes and hair highlights.

### 2. Photographic Modifiers
* **Octabox (150cm):** Double-diffused, ultra-soft wrap, low contrast (classic beauty & catalogue).
* **Beauty Dish (Silver):** Center deflector plate; punchy core, soft feathered falloff, crisp round iris catchlights.
* **Fresnel Spotlight:** Stepped optical glass; hard collimated beam, razor chiaroscuro shadows (film noir & cinema).
* **Hard Snoot:** Narrow 15° honeycomb cone; dramatic isolation pool of light.
* **Direct On-Camera Flash:** Raw lens-axis strobe; hard pop, stark micro-shadows, fashion snapshot aesthetic.
* **Stripbox:** Tall narrow kicker; razor-thin edge accents for jawlines and shoulders.

### 3. Kelvin Color Science
* $2500\text{K} - 3200\text{K}$: Warm candle & studio tungsten halogen.
* $4800\text{K} - 5500\text{K}$: Neutral daylight strobe.
* $6500\text{K} - 8000\text{K}$: Cool overcast daylight & cinematic skylight.

### 4. Built-in Studio Presets (JSON-Expandable)
* **`Rembrandt Classic`**: 45° soft key, balanced fill, subtle rim separation.
* **`High-Fashion Beauty`**: Elevated beauty dish, subtle floor fill, dual kickers.
* **`Direct On-Camera Flash`**: Raw lens-axis flash snapshot, zero fill, dark falloff.
* **`Cinematic Film Noir`**: Hard 75° fresnel side key, pitch-black shadows, razor rim.
* **`Golden Hour Window`**: Warm 3000K low-angle sunlight beam, cool 6500K sky fill.

---

## Installation

Inside your ComfyUI directory:

```bash
cd custom_nodes
git clone https://github.com/maciejtajber/ComfyUI-StudioLight.git
```

Restart ComfyUI. Find the node under:  
**`lighting` → `Studio Light Rig (3-Point Photographic Console)`**

---

## Author & License

Built by **[Maciej Tajber (MATA)](https://github.com/maciejtajber)**.  
MIT License — free for commercial and personal production.
