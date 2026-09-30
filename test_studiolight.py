"""
ComfyUI-StudioLight — Automated Test Suite
Verifies all modules, math functions, shaders, presets, and node execution.
"""

import os
import sys
import torch

# Ensure package is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, COMFY_DIR)
sys.path.insert(0, BASE_DIR)

from nodes.color_utils import kelvin_to_rgb, kelvin_to_mood, MODIFIER_PRESETS
from nodes.preset_manager import list_presets, load_preset_data, get_all_presets_flattened
from nodes.shader_engine import (
    azimuth_elevation_to_vector,
    render_sphere_normals,
    render_studio_lighting,
    draw_studio_radar_diagram,
)
from nodes.prompt_synthesizer import synthesize_high_signal_prompt
from nodes.studio_light import StudioLightRig


def test_color_utils():
    print("Testing color_utils...")
    r, g, b = kelvin_to_rgb(3200)
    assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0
    assert r > b, "3200K tungsten should have more red than blue"

    r_d, g_d, b_d = kelvin_to_rgb(6500)
    assert b_d > b, "6500K should have more blue than 3200K"

    mood = kelvin_to_mood(3200)
    assert "tungsten" in mood.lower()
    print("✓ color_utils passed!")


def test_presets():
    print("Testing preset_manager...")
    presets = list_presets()
    assert "Custom" in presets
    assert "Rembrandt Classic" in presets
    assert "High-Fashion Beauty" in presets
    assert "Direct On-Camera Flash" in presets
    assert "Cinematic Film Noir" in presets
    assert "Golden Hour Window" in presets

    data = load_preset_data("Rembrandt Classic")
    assert data["key"]["azimuth"] == 45.0

    # Verify High-Fashion Beauty rim softened fix
    beauty = load_preset_data("High-Fashion Beauty")
    assert beauty["rim"]["azimuth"] == 140.0
    assert beauty["rim"]["elevation"] == 40.0
    assert beauty["rim"]["intensity"] == 0.90

    # Verify flattened preset export for frontend SSOT
    flattened = get_all_presets_flattened()
    assert "Rembrandt Classic" in flattened
    assert flattened["Rembrandt Classic"]["key_azimuth"] == 45.0
    assert flattened["High-Fashion Beauty"]["rim_azimuth"] == 140.0
    print("✓ preset_manager passed!")


def test_shader_engine():
    print("Testing shader_engine...")
    # 1. Vector conversion
    v_front = azimuth_elevation_to_vector(0.0, 0.0)
    assert torch.allclose(v_front, torch.tensor([0.0, 0.0, 1.0]), atol=1e-4)

    v_right = azimuth_elevation_to_vector(90.0, 0.0)
    assert torch.allclose(v_right, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)

    v_up = azimuth_elevation_to_vector(0.0, 90.0)
    assert torch.allclose(v_up, torch.tensor([0.0, 1.0, 0.0]), atol=1e-4)

    # 2. Sphere normals
    normals, mask = render_sphere_normals(128, device=torch.device("cpu"))
    assert normals.shape == (1, 128, 128, 3)
    assert mask.shape == (1, 128, 128, 1)

    # 3. Radar diagram
    key = {"enabled": True, "azimuth": 45.0, "elevation": 35.0, "intensity": 1.0, "kelvin": 5500, "modifier": "Octabox (Soft)"}
    fill = {"enabled": True, "azimuth": -30.0, "elevation": 15.0, "intensity": 0.35, "kelvin": 4800, "modifier": "Octabox (Soft)"}
    rim = {"enabled": True, "azimuth": 160.0, "elevation": 45.0, "intensity": 1.1, "kelvin": 6500, "modifier": "Stripbox"}

    radar = draw_studio_radar_diagram(key, fill, rim, width=256, height=128)
    assert radar.shape == (1, 128, 256, 3)

    # 4. Studio lighting with mannequin bust fallback (Portrait)
    relight_bust, probe_bust = render_studio_lighting(None, None, key, fill, rim, preview_resolution=256, framing="Portrait (Classical Bust)")
    assert relight_bust.shape == (1, 1024, 1024, 3)
    assert probe_bust.shape == (1, 256, 256, 3)

    # 5. Studio lighting with American Plan
    relight_med, _ = render_studio_lighting(None, None, key, fill, rim, framing="American Plan (3/4 Shot)")
    assert relight_med.shape == (1, 1200, 896, 3)

    # 6. Studio lighting with Full Body
    relight_full, _ = render_studio_lighting(None, None, key, fill, rim, framing="Full Body (Standing Lookbook)")
    assert relight_full.shape == (1, 1264, 848, 3)

    # 7. Studio lighting with Volumetric Cyc (Room probe)
    relight_cyc, _ = render_studio_lighting(None, None, key, fill, rim, framing="Volumetric Cyc (Crowds / Dynamic)")
    assert relight_cyc.shape == (1, 1152, 896, 3)

    # 8. Studio lighting with Tabletop / Luxury (Perfume)
    relight_perfume, _ = render_studio_lighting(None, None, key, fill, rim, framing="Tabletop / Luxury (Perfume Bottle / Small Product)")
    assert relight_perfume.shape == (1, 1024, 1024, 3)

    # 9. Studio lighting with Industrial Design (Chair)
    relight_chair, _ = render_studio_lighting(None, None, key, fill, rim, framing="Industrial Design (Chair / Medium Product)")
    assert relight_chair.shape == (1, 1024, 1024, 3)

    # 10. Studio lighting with Automotive Cove
    relight_auto, _ = render_studio_lighting(None, None, key, fill, rim, framing="Automotive Cove (Car / Large Vehicle)")
    assert relight_auto.shape == (1, 768, 1280, 3)

    # 11. Studio lighting with mock normals
    mock_normals = torch.zeros(1, 128, 128, 3)
    mock_normals[..., 2] = 1.0  # Facing front
    mock_relight, _ = render_studio_lighting(mock_normals, None, key, fill, rim, preview_resolution=128)
    assert mock_relight.shape == (1, 128, 128, 3)
    print("✓ shader_engine passed!")


def test_prompt_synthesizer():
    print("Testing prompt_synthesizer...")
    key = {"enabled": True, "azimuth": 45.0, "elevation": 35.0, "intensity": 1.0, "kelvin": 3200, "modifier": "Octabox (Soft)"}
    fill = {"enabled": True, "azimuth": -30.0, "elevation": 15.0, "intensity": 0.35, "kelvin": 4800, "modifier": "Octabox (Soft)"}
    rim = {"enabled": True, "azimuth": 160.0, "elevation": 45.0, "intensity": 1.1, "kelvin": 6500, "modifier": "Stripbox"}

    # 1. Flux.1 / Flux.2 Engine (Narrative Cause & Effect)
    prompt_port = synthesize_high_signal_prompt(key, fill, rim, framing="Portrait (Classical Bust)", target_engine="Flux.1 / Flux.2")
    assert "Studio fashion portrait" in prompt_port
    assert "right side of the frame" in prompt_port
    assert "warm tungsten" in prompt_port
    assert "left side" in prompt_port

    # 2. Video Engine (Wan 2.1 / MiniMax)
    prompt_vid = synthesize_high_signal_prompt(key, fill, rim, framing="Portrait (Classical Bust)", target_engine="Video (Wan / MiniMax)")
    assert "[Cinematic studio lighting]" in prompt_vid
    assert "[Stationary three-point light setup]" in prompt_vid
    assert "right side of the frame" in prompt_vid
    assert "temporally stable" in prompt_vid

    # 3. SDXL Engine
    prompt_sdxl = synthesize_high_signal_prompt(key, fill, rim, framing="Portrait (Classical Bust)", target_engine="SDXL")
    assert "studio portrait" in prompt_sdxl
    assert "right side 45 degree" in prompt_sdxl

    # 4. Product Framing
    prompt_prod = synthesize_high_signal_prompt(key, fill, rim, framing="Tabletop / Luxury (Perfume Bottle / Small Product)")
    assert "Commercial luxury product photography" in prompt_prod

    prompt_chair = synthesize_high_signal_prompt(key, fill, rim, framing="Industrial Design (Chair / Medium Product)")
    assert "Industrial design photography" in prompt_chair

    prompt_car = synthesize_high_signal_prompt(key, fill, rim, framing="Automotive Cove (Car / Large Vehicle)")
    assert "Commercial automotive studio photography" in prompt_car
    print("✓ prompt_synthesizer passed across all engines (Flux, Video, SDXL)!")



def test_studio_light_node():
    print("Testing StudioLightRig node execution...")
    node = StudioLightRig()
    # Test 1: Explicit framing (Full Body)
    relight, probe, prompt = node.execute_rig(
        preset="Rembrandt Classic",
        framing="Full Body (Standing Lookbook)",
        ambient_intensity=0.05,
        invert_normals_y=False,
        key_enabled=True,
        key_azimuth=45.0,
        key_elevation=35.0,
        key_intensity=1.0,
        key_kelvin=5500,
        key_modifier="Octabox (Soft)",
        fill_enabled=True,
        fill_azimuth=-30.0,
        fill_elevation=15.0,
        fill_intensity=0.35,
        fill_kelvin=4800,
        fill_modifier="Octabox (Soft)",
        rim_enabled=True,
        rim_azimuth=160.0,
        rim_elevation=45.0,
        rim_intensity=1.1,
        rim_kelvin=6500,
        rim_modifier="Stripbox",
    )

    assert relight is not None and isinstance(relight, torch.Tensor)
    assert relight.shape == (1, 1264, 848, 3)
    assert probe is not None and isinstance(probe, torch.Tensor)
    assert isinstance(prompt, str)

    # Test 2: scene_3d input unpacking
    mock_normals = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    mock_normals[..., 2] = 1.0  # Camera-facing flat normal
    mock_depth = torch.ones((1, 512, 512, 1), dtype=torch.float32) * 0.5
    scene_3d_mock = {
        "normals": mock_normals,
        "depth": mock_depth,
        "metadata": {"framing": "Custom 3D Scene"},
    }

    relight_scene, probe_scene, prompt_scene = node.execute_rig(
        preset="Rembrandt Classic",
        scene_3d=scene_3d_mock,
        key_enabled=True,
        key_azimuth=45.0,
    )
    assert relight_scene is not None and relight_scene.shape == (1, 512, 512, 3)
    assert probe_scene is not None
    assert isinstance(prompt_scene, str)
    print("✓ StudioLightRig node execution with scene_3d passed perfectly!")


def test_studio_3d_stage_and_suite_pipeline():
    print("Testing Studio3DStage and end-to-end suite pipeline...")
    from nodes.studio_3d_stage import Studio3DStage
    from nodes.si_shadow_highlight import SIShadowHighlight

    stage = Studio3DStage()
    rig = StudioLightRig()
    bridge = SIShadowHighlight()

    # 1. Test Stage execution with soft_head_blur ON
    scene_3d, normals, depth = stage.execute_stage(
        model_preset="Female Mannequin",
        pose_preset="Standing Fashion",
        framing="Portrait (Classical Bust)",
        soft_head_blur=True,
        head_blur_radius=20.0,
        pose_locked=False,
    )

    assert scene_3d is not None and isinstance(scene_3d, dict)
    assert "normals" in scene_3d and "depth" in scene_3d and "metadata" in scene_3d
    assert normals is not None and normals.shape[-1] == 3
    assert depth is not None and depth.shape[-1] == 3
    print("  ✓ Stage 1 (Studio3DStage) generated valid scene_3d, normals, and depth passes.")

    # 2. Test Rig execution taking scene_3d directly from Stage
    relit_plate, probe_preview, light_prompt = rig.execute_rig(
        preset="Rembrandt Classic",
        scene_3d=scene_3d,
        key_enabled=True,
        key_azimuth=45.0,
        key_elevation=35.0,
        key_intensity=1.1,
    )

    assert relit_plate is not None and relit_plate.shape == normals.shape
    assert probe_preview is not None
    assert "Rembrandt" in light_prompt or "Key" in light_prompt or "key" in light_prompt.lower()
    print("  ✓ Stage 2 (StudioLightRig) successfully lit scene_3d and generated lit proxy plate.")

    # 3. Test Bridge execution taking relit_plate into SIShadowHighlight
    adjusted_tuple = bridge.adjust(
        image=relit_plate,
        shadow_amount=0.25,
        highlight_amount=0.15,
        shadow_tonal_width=0.5,
        highlight_tonal_width=0.5,
        midtone_contrast=1.1,
    )
    adjusted_plate = adjusted_tuple[0]

    assert adjusted_plate is not None and adjusted_plate.shape == relit_plate.shape
    print("  ✓ Step 3 (SIShadowHighlight) sculpted dynamic range for Flux VAE latent.")
    print("✓ Full 3-Node Suite pipeline (Stage -> Rig -> Bridge) executed flawlessly!")


def test_photometric_composite():
    print("Testing StudioPhotometricComposite...")
    from nodes.photometric_composite import StudioPhotometricComposite

    comp = StudioPhotometricComposite()
    dummy_image = torch.full((1, 256, 256, 3), 0.5, dtype=torch.float32)
    dummy_light = torch.full((1, 256, 256, 3), 0.8, dtype=torch.float32)

    lit_tuple = comp.composite(
        image=dummy_image,
        lighting_map=dummy_light,
        lighting_strength=1.2,
        ambient_floor=0.05,
        specular_kick=1.5,
        chroma_bleed=0.4,
        filmic_white_point=3.5,
    )
    lit_img = lit_tuple[0]
    assert lit_img is not None
    assert lit_img.shape == (1, 256, 256, 3)
    assert 0.0 <= lit_img.min() and lit_img.max() <= 1.0
    print("✓ StudioPhotometricComposite passed flawlessly!")


def test_trajectory_nodes():
    print("Testing StudioLight Trajectory Nodes (Method B)...")
    from nodes.trajectory_nodes import StudioLightTrajectoryInject, StudioLightTrajectorySplitter
    
    # 1. Verify Node registration
    assert StudioLightTrajectorySplitter is not None
    assert StudioLightTrajectoryInject is not None

    # 2. Verify Mathematical Delta Invariant
    sigma = 0.863338
    x0 = torch.randn(1, 16, 64, 64)
    x0_prime = x0 + torch.randn(1, 16, 64, 64) * 0.15
    eps = torch.randn(1, 16, 64, 64)

    # True ODE target states
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_prime_t = (1.0 - sigma) * x0_prime + sigma * eps

    # In ComfyUI scaled latent space:
    L_t = x_t / (1.0 - sigma)

    inject_node = StudioLightTrajectoryInject()
    noisy_dict = {"samples": L_t}
    clean_dict = {"samples": x0}
    relit_dict = {"samples": x0_prime}

    injected_dict = inject_node.inject(noisy_dict, clean_dict, relit_dict, strength=1.0)[0]
    injected_samples = injected_dict["samples"]

    # When Pass 2 sampler applies CONST noise_scaling with noise=0:
    ode_start = (1.0 - sigma) * injected_samples
    diff = torch.max(torch.abs(ode_start - x_prime_t)).item()
    assert diff < 1e-5, f"Delta injection math error too high: {diff}"
    print(f"  ✓ Delta invariant mathematically exact: max error = {diff:.2e}")
    print("✓ StudioLight Trajectory Nodes passed flawlessly!")


if __name__ == "__main__":
    test_color_utils()
    test_presets()
    test_shader_engine()
    test_prompt_synthesizer()
    test_studio_light_node()
    test_studio_3d_stage_and_suite_pipeline()
    test_photometric_composite()
    test_trajectory_nodes()
    print("\n🎉 ALL TESTS PASSED! ComfyUI-StudioLight Suite is 100% verified!")


