"""
ComfyUI-StudioLight — Extended 3-Domain Photometric Validation Suite
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL

Runs physical and semantic validation across:
  1. Night Scene Stress Test: Mathematical proof against squaring darkness.
  2. Golden Hour Chromaticity Wrap: Ambient environment color integration.
  3. 5-Domain Prompt Matrix Battery: Complete decoupling & suffix correctness.
"""

import unittest
import torch

from nodes.studio_light import StudioLightRig
from nodes.photometric_composite import (
    StudioPhotometricComposite,
    measure_photometric_environment,
)
from nodes.prompt_sanitizer import StudioPromptSanitizer, detect_photometric_exposure


def test_night_scene_resilience():
    # Synthetic low-key night city scene
    H, W = 128, 128
    night_bg = torch.full((1, H, W, 3), 0.08, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, H), torch.linspace(-1.0, 1.0, W), indexing="ij")
    r2 = torch.clamp(xx**2 + yy**2, 0.0, 1.0)
    subj_mask = (r2 < 0.35).float().unsqueeze(0).unsqueeze(-1)
    night_plate = night_bg * (1.0 - subj_mask) + torch.tensor([0.15, 0.12, 0.10]).view(1, 1, 1, 3) * subj_mask

    normals = torch.zeros(1, H, W, 3)
    normals[..., 2] = 1.0
    depth = torch.zeros(1, H, W, 1)
    depth[..., 0] = torch.where(r2 < 0.35, 0.85, 0.10)

    rig = StudioLightRig()
    relight, _, _, _ = rig.execute_rig(
        preset="Custom",
        key_enabled=True, key_azimuth=45.0, key_elevation=30.0, key_intensity=1.5, key_kelvin=5500,
        fill_enabled=False,
        rim_enabled=True, rim_azimuth=-150.0, rim_elevation=20.0, rim_intensity=1.0, rim_kelvin=3200,
        normals_optional=normals, depth_optional=depth,
    )

    comp = StudioPhotometricComposite()
    lit_night = comp.composite(
        image=night_plate,
        lighting_map=relight,
        lighting_strength=1.0,
        wall_shadow_density=0.65,
        bg_exposure_ev=0.0,
    )[0]

    bg_in = night_plate[0, 0:20, 0:20, :].mean().item()
    bg_out = lit_night[0, 0:20, 0:20, :].mean().item()
    ratio = bg_out / bg_in
    assert abs(ratio - 1.0) < 0.05, f"Night background exposure altered! ratio={ratio:.4f}"


def test_golden_hour_chromaticity():
    H, W = 64, 64
    sunset_bg = torch.tensor([0.65, 0.40, 0.15]).view(1, 1, 1, 3).repeat(1, H, W, 1)
    d_norm = torch.full((1, H, W, 1), 0.05)
    m_fg = torch.zeros(1, H, W, 1)

    l_env, c_env, m_sky, *rest = measure_photometric_environment(sunset_bg, d_norm, d_split=0.35, m_fg=m_fg)
    r_to_b = float(c_env[0, 0, 0, 0] / c_env[0, 0, 0, 2])
    assert r_to_b > 1.30, f"Light meter failed to capture warm golden hour tint! R/B={r_to_b:.3f}"


def test_five_domain_prompt_battery():
    test_cases = [
        (
            "Universal Outdoor (Paris)",
            "A model in a glossy patent leather coat walking past neoclassical Parisian architecture on a bright sunny day",
            "universal",
            "diffuse even ambient illumination",
        ),
        (
            "Low-Key Night Street",
            "Editorial fashion photography of a woman in a liquid chrome dress on a rainy night in Tokyo with neon reflections",
            "low_key",
            "diffuse low-key ambient illumination",
        ),
        (
            "Warm Sunset / Golden Hour",
            "Romantic sunset portrait on a Mediterranean terrace with warm golden hour sky, wearing glossy amber jewelry",
            "warm_ambient",
            "soft diffused warm ambient fill",
        ),
        (
            "Universal Interior",
            "High-end luxury portrait inside a brutalist art gallery loft with polished concrete floor and window bounce",
            "universal",
            "diffuse even ambient illumination",
        ),
        (
            "Explicit Studio Cyclorama",
            "High fashion beauty portrait against a seamless grey paper cyclorama backdrop with dramatic rim lighting",
            "studio_explicit",
            "diffuse even studio ambient illumination",
        ),
        (
            "Non-Studio (Airplane)",
            "A passenger looking out the window inside a commercial airplane cabin",
            "universal",
            "diffuse even ambient illumination",
        ),
        (
            "Non-Studio (Garage)",
            "A mechanic with dirty hands standing next to a race car inside an industrial garage",
            "universal",
            "diffuse even ambient illumination",
        ),
    ]

    sanitizer = StudioPromptSanitizer()

    for category, prompt, exp_exposure, exp_kw in test_cases:
        detected = detect_photometric_exposure(prompt)
        p1_v2, p2_v2 = sanitizer.sanitize(prompt, lighting_policy="RIG_OVERRIDE")
        p1_log, p2_raw = sanitizer.sanitize(prompt, lighting_policy="VERBATIM_RAW", pass1_log_acquisition=True)
        p1_leg, _ = sanitizer.sanitize(prompt, lighting_policy="VERBATIM_RAW", pass1_log_acquisition=False)

        assert detected == exp_exposure, f"Exposure mode mismatch on '{category}': got '{detected}', expected '{exp_exposure}'"
        assert exp_kw in p1_v2 and exp_kw in p1_log and exp_kw in p1_leg
        assert "neutral diffuse reference capture" in p1_v2.lower()
        assert "logarithmic sensor capture" in p1_log.lower()
        assert p2_raw == prompt

        if "rim lighting" in prompt.lower():
            assert "rim light" not in p2_v2.lower()

        if exp_exposure != "studio_explicit":
            assert "studio" not in p1_v2.lower() and "cyclorama" not in p1_v2.lower()
            assert "studio" not in p1_leg.lower() and "cyclorama" not in p1_leg.lower()

        for spec_token in ["glossy", "patent", "liquid chrome"]:
            if spec_token in prompt.lower():
                assert spec_token not in p1_leg.lower()
                assert spec_token in p1_v2.lower()
                assert spec_token in p1_log.lower()


class Extended3DomainTests(unittest.TestCase):
    def test_night_scene_resilience(self):
        test_night_scene_resilience()

    def test_golden_hour_chromaticity(self):
        test_golden_hour_chromaticity()

    def test_five_domain_prompt_battery(self):
        test_five_domain_prompt_battery()


if __name__ == "__main__":
    unittest.main()
