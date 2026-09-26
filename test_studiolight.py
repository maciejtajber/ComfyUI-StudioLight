"""
ComfyUI-StudioLight — Automated Test Suite
Verifies all modules, math functions, shaders, presets, and node execution.
"""

import os
import sys
import torch

# Ensure package is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    relight, probe, cond, prompt, comp = node.execute_rig(
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
        start_step=0.40,
        end_step=1.00,
        conditioning_strength=1.0,
        weight_type="A_NORM",
    )

    assert relight is not None and isinstance(relight, torch.Tensor)
    assert relight.shape == (1, 1264, 848, 3)
    assert probe is not None and isinstance(probe, torch.Tensor)
    assert isinstance(prompt, str)
    assert comp is not None and comp.shape == relight.shape

    # Test 2: Auto detection (Prompt mentions runway / models -> Cyc)
    relight_auto, _, _, _, _ = node.execute_rig(
        preset="Rembrandt Classic",
        framing="Auto (Detect from Prompt)",
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
        start_step=0.40,
        end_step=1.00,
        conditioning_strength=1.0,
        weight_type="A_NORM",
        prompt_text_optional="three fashion models walking together on a runway presentation",
    )
    assert relight_auto.shape == (1, 1152, 896, 3)

    # Test 3: Photometric Composite with Draft Image
    draft_mock = torch.full((1, 512, 512, 3), 0.7)
    _, _, _, _, lit_out = node.execute_rig(
        preset="Rembrandt Classic",
        framing="Portrait (Classical Bust)",
        key_enabled=True,
        key_azimuth=45.0,
        image_optional=draft_mock,
    )
    assert lit_out is not None and isinstance(lit_out, torch.Tensor)
    assert lit_out.shape == (1, 512, 512, 3)
    # Verify directional contrast on the cheeks (y=200, x=300 right, x=200 left in 512x512)
    right_lum = lit_out[0, 200, 300, :].mean().item()
    left_lum = lit_out[0, 200, 200, :].mean().item()
    assert right_lum > left_lum, f"Key at +45 should make right ({right_lum:.2f}) brighter than left ({left_lum:.2f})"
    print("✓ StudioLightRig node execution and photometric composite passed perfectly!")


if __name__ == "__main__":
    test_color_utils()
    test_presets()
    test_shader_engine()
    test_prompt_synthesizer()
    test_studio_light_node()
    print("\n🎉 ALL TESTS PASSED! ComfyUI-StudioLight is 100% verified!")
