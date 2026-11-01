"""Integration test for shipped workflows and directional photometric truth.

Enforces the 2026-10-31 Invariant:
1. Workflows MUST serialize Node #14 with 'marigold_native'.
2. Workflows MUST serialize Node #15 with invert_normals_x=False and invert_normals_y=False.
3. A Key light at Azimuth +75° (camera right) MUST produce higher irradiance
   on a surface with normal +X than on a surface with normal -X.
"""
import json
from pathlib import Path
import torch

from nodes.studio_light import StudioLightRig
from nodes.shader_engine import render_studio_lighting

WORKFLOWS_DIR = Path(__file__).resolve().parent.parent / "workflows"


def test_workflow_coordinate_invariants():
    """Verify serialized settings in shipped workflow JSONs."""
    workflows = ["StudioLight_Create.json", "StudioLight_Relight.json"]
    for wf_name in workflows:
        wf_path = WORKFLOWS_DIR / wf_name
        assert wf_path.is_file(), f"Workflow {wf_name} missing from {WORKFLOWS_DIR}"

        with open(wf_path) as f:
            data = json.load(f)

        found_normals = False
        found_rig = False
        found_contract = False

        for n in data.get("nodes", []):
            if n.get("type") == "StudioLightMarigoldNormalsTest":
                found_normals = True
                convention = n.get("widgets_values", [])[5]
                assert convention == "marigold_native", (
                    f"In {wf_name}, Node #14 output_convention must be 'marigold_native', got {convention}"
                )

            if n.get("type") == "StudioLightRig":
                found_rig = True
                wv = n.get("widgets_values", [])
                inv_y = wv[2]
                inv_x = wv[26]
                assert inv_x is False, (
                    f"In {wf_name}, StudioLightRig invert_normals_x must be False, got {inv_x}"
                )
                assert inv_y is False, (
                    f"In {wf_name}, StudioLightRig invert_normals_y must be False, got {inv_y}"
                )
                assert len(wv) >= 31, (
                    f"In {wf_name}, StudioLightRig must have at least 31 widget values, got {len(wv)}"
                )

            if n.get("type") == "StudioLightSceneContract":
                found_contract = True
                wv = n.get("widgets_values", [])
                assert len(wv) >= 5, (
                    f"In {wf_name}, StudioLightSceneContract must have at least 5 widget values, got {len(wv)}"
                )
                assert wv[3] == "Auto", (
                    f"In {wf_name}, StudioLightSceneContract refinement_mode must default to 'Auto', got {wv[3]}"
                )

        assert found_normals, f"Node #14 missing in {wf_name}"
        assert found_rig, f"Node #15 missing in {wf_name}"
        assert found_contract, f"StudioLightSceneContract missing in {wf_name}"


def test_directional_photometric_truth():
    """Verify that light from camera right (+75°) illuminates +X surface, not -X."""
    H, W = 64, 64
    # Left half: surface normal points LEFT (-X)
    # Right half: surface normal points RIGHT (+X)
    normals = torch.zeros(1, H, W, 3, dtype=torch.float32)
    normals[:, :, : W // 2, 0] = -1.0  # Left facing
    normals[:, :, : W // 2, 2] = 0.2   # Slight forward tilt
    normals[:, :, W // 2 :, 0] = 1.0   # Right facing
    normals[:, :, W // 2 :, 2] = 0.2   # Slight forward tilt
    normals = torch.nn.functional.normalize(normals, dim=-1)

    # Encode as ComfyUI [0, 1] RGB
    normals_rgb = (normals + 1.0) * 0.5

    key = {
        "enabled": True,
        "azimuth": 75.0,  # Camera right
        "elevation": 20.0,
        "distance": 1.8,
        "intensity": 1.5,
        "kelvin": 3200,
        "modifier": "Fresnel Spotlight",
    }
    fill = {"enabled": False, "azimuth": -45.0, "elevation": 0.0, "distance": 2.5, "intensity": 0.0, "kelvin": 3200, "modifier": "Octabox (Soft)"}
    rim = {"enabled": False, "azimuth": -150.0, "elevation": 35.0, "distance": 1.5, "intensity": 0.0, "kelvin": 6500, "modifier": "Stripbox"}

    # 1. Correct configuration: invert_normals_x=False, invert_normals_y=False
    lit_map, _ = render_studio_lighting(
        normals=normals_rgb,
        depth=None,
        key_params=key,
        fill_params=fill,
        rim_params=rim,
        invert_normals_x=False,
        invert_normals_y=False,
        _render_probe=False,
    )

    left_half_energy = lit_map[:, :, : W // 2, :].mean().item()
    right_half_energy = lit_map[:, :, W // 2 :, :].mean().item()

    assert right_half_energy > 5.0 * left_half_energy, (
        f"Directional failure: right half ({right_half_energy:.4f}) must be significantly brighter "
        f"than left half ({left_half_energy:.4f}) when Key is at +75°!"
    )

    # 2. Assert that old buggy configuration (invert_normals_x=True) inverted this!
    buggy_lit_map, _ = render_studio_lighting(
        normals=normals_rgb,
        depth=None,
        key_params=key,
        fill_params=fill,
        rim_params=rim,
        invert_normals_x=True,
        invert_normals_y=False,
        _render_probe=False,
    )
    buggy_left = buggy_lit_map[:, :, : W // 2, :].mean().item()
    buggy_right = buggy_lit_map[:, :, W // 2 :, :].mean().item()

    assert buggy_left > buggy_right, "Invariant assertion: invert_normals_x=True flipped left/right."


if __name__ == "__main__":
    test_workflow_coordinate_invariants()
    test_directional_photometric_truth()
    print("✓ All workflow coordinate & directional photometric invariants PASSED.")
