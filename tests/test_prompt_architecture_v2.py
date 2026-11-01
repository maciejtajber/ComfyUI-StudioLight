"""
ComfyUI-StudioLight — Prompt & Albedo Architecture v2 Test Suite
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL

Validates the v2 semantic parser, lighting policies (RIG_OVERRIDE vs HYBRID_PRACTICAL),
material preservation, practical emitter extraction, and Pass 2 prompt cleanliness.
"""

import unittest
from nodes.semantic_parser import (
    parse_scene_description,
    build_pass1_prompt_v2,
    build_pass2_prompt_v2,
    MaterialCategory,
    LightingPolicy,
    PASS1_NEUTRAL_DIFFUSE_CAPTURE,
    PASS2_MINIMAL_REFINEMENT,
)
from nodes.prompt_sanitizer import StudioPromptSanitizer
from nodes.consistent_relighting import StudioLightSceneContract
import json


class TestPromptArchitectureV2(unittest.TestCase):

    def setUp(self):
        self.sanitizer = StudioPromptSanitizer()

    def test_mandatory_example_1_crying_woman_with_phone(self):
        """
        Original: A woman crying in darkness, tears running down her cheeks, holding a smartphone illuminating her face.
        Pass 1: Must preserve tears, crying, phone, but neutralize darkness and screen illumination.
        Pass 2: Under RIG_OVERRIDE, must preserve tears, crying, phone, without directional illumination directives.
        """
        raw = "A woman crying in darkness, tears running down her cheeks, holding a smartphone illuminating her face."
        scene = parse_scene_description(raw, LightingPolicy.RIG_OVERRIDE.value)
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="RIG_OVERRIDE")

        # Semantic content preservation
        self.assertIn("crying", p1.lower())
        self.assertIn("tears running down her cheeks", p1.lower())
        self.assertIn("smartphone", p1.lower())
        self.assertIn("crying", p2.lower())
        self.assertIn("tears running down her cheeks", p2.lower())
        self.assertIn("smartphone", p2.lower())

        # Directional lighting neutralization
        self.assertNotIn("in darkness", p1.lower())
        self.assertNotIn("illuminating her face", p1.lower())
        self.assertNotIn("in darkness", p2.lower())
        self.assertNotIn("illuminating her face", p2.lower())

        # Neutral capture guidance
        self.assertIn("neutral diffuse reference capture", p1.lower())
        self.assertIn("preserving subject identity", p2.lower())

    def test_mandatory_example_2_chrome_android(self):
        """
        Original: A black chrome android illuminated by harsh red rim lighting.
        Must preserve black chrome android and metallic material identity.
        Must strip harsh red rim lighting from Pass 1 and Pass 2 under RIG_OVERRIDE.
        """
        raw = "A black chrome android illuminated by harsh red rim lighting."
        scene = parse_scene_description(raw, LightingPolicy.RIG_OVERRIDE.value)
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="RIG_OVERRIDE")

        self.assertIn(MaterialCategory.METALLIC.value, scene.material_categories)
        self.assertIn("chrome", p1.lower())
        self.assertIn("reflective metallic", p1.lower())
        self.assertIn("chrome", p2.lower())
        self.assertIn("reflective metallic", p2.lower())

        self.assertNotIn("rim light", p1.lower())
        self.assertNotIn("rim light", p2.lower())
        self.assertNotIn("harsh red", p1.lower())
        self.assertNotIn("harsh red", p2.lower())

    def test_mandatory_example_3_luxury_handbag_on_marble(self):
        """
        Original: A black handbag resting on a polished marble table under harsh spotlight illumination.
        Must preserve handbag, polished marble table, and glossy/dielectric material identity.
        Must strip spotlight from Pass 1 and Pass 2 under RIG_OVERRIDE.
        """
        raw = "A black handbag resting on a polished marble table under harsh spotlight illumination."
        scene = parse_scene_description(raw, LightingPolicy.RIG_OVERRIDE.value)
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="RIG_OVERRIDE")

        self.assertIn(MaterialCategory.DIELECTRIC_GLOSSY.value, scene.material_categories)
        self.assertIn("black handbag", p1.lower())
        self.assertIn("polished marble table", p1.lower())
        self.assertIn("black handbag", p2.lower())
        self.assertIn("polished marble table", p2.lower())

        self.assertNotIn("spotlight", p1.lower())
        self.assertNotIn("spotlight", p2.lower())

    def test_techno_club_flash_and_cast_shadows(self):
        """
        Original: A stylish young techno girl in an underground dark techno club, illuminated by a harsh on-camera flash,
        deep dark shadows casting on the concrete wall, wet shiny latex jacket, sweat on skin, 35mm film photography.
        Must preserve: young techno girl, underground techno club, wet shiny latex jacket, sweat on skin, 35mm film photography.
        Must strip: on-camera flash, cast shadows.
        """
        raw = (
            "A stylish young techno girl in an underground dark techno club, illuminated by a harsh on-camera flash, "
            "deep dark shadows casting on the concrete wall, wet shiny latex jacket, sweat on skin, 35mm film photography."
        )
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="RIG_OVERRIDE")

        self.assertIn("techno girl", p1.lower())
        self.assertIn("latex jacket", p1.lower())
        self.assertIn("sweat on skin", p1.lower())
        self.assertIn("35mm film photography", p1.lower())

        self.assertNotIn("on-camera flash", p1.lower())
        self.assertNotIn("on-camera flash", p2.lower())
        self.assertNotIn("shadows casting on", p1.lower())
        self.assertNotIn("shadows casting on", p2.lower())
        # Zero orphan grammar
        self.assertNotIn("illuminated by a harsh", p1.lower())
        self.assertNotIn("illuminated by a harsh", p2.lower())

    def test_hybrid_practical_preserves_emissive_glow(self):
        """
        In HYBRID_PRACTICAL mode:
        A practical light source (like a candle or glowing screen) retains its in-scene emissive description
        for Pass 2, but directional studio lighting directives (like flash or rim) are still neutralized.
        """
        raw = "A monk holding a burning candle in darkness, chiaroscuro lighting, ancient monastery."
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="HYBRID_PRACTICAL")

        self.assertIn("candle", p1.lower())
        self.assertIn("candle", p2.lower())
        self.assertIn("monk", p2.lower())
        self.assertIn("monastery", p2.lower())

        # Chiaroscuro stripped
        self.assertNotIn("chiaroscuro", p1.lower())
        self.assertNotIn("chiaroscuro", p2.lower())

    def test_verbatim_raw_backward_compatibility(self):
        """
        When lighting_policy is set to VERBATIM_RAW, Pass 2 must return the exact master prompt untouched.
        """
        raw = "High fashion editorial portrait, harsh spotlight, stark shadows."
        p1, p2 = self.sanitizer.sanitize(raw, lighting_policy="VERBATIM_RAW")
        self.assertEqual(p2, raw)
        self.assertNotIn("spotlight", p1.lower())

    def test_pass2_refinement_clause_does_not_contain_fine_surface_details(self):
        """
        PASS2_MINIMAL_REFINEMENT must not contain 'fine surface details' (the FLUX pimple trigger).
        """
        self.assertNotIn("fine surface details", PASS2_MINIMAL_REFINEMENT.lower())
        self.assertIn("preserving subject identity, composition, and material characteristics", PASS2_MINIMAL_REFINEMENT)

    def test_scene_contract_auto_mode(self):
        """
        In Auto mode, StudioLightSceneContract builds clean prompts without 'fine surface details'.
        """
        node = StudioLightSceneContract()
        contract, p1, p2, report = node.build(
            master_prompt="portrait of a model",
            environment="Auto",
            ambient_intent="Auto",
            refinement_mode="Auto",
            custom_pass2_prompt="",
        )
        self.assertNotIn("fine surface details", p2.lower())
        self.assertIn("preserving subject identity", p2.lower())
        self.assertEqual(contract["refinement_mode"], "Auto")
        self.assertEqual(contract["effective_pass2_prompt"], p2)

        rep = json.loads(report)
        self.assertEqual(rep["refinement_mode"], "Auto")
        self.assertEqual(rep["effective_pass2_prompt"], p2)

    def test_scene_contract_custom_override_mode(self):
        """
        In Custom Override mode, custom_pass2_prompt is delivered verbatim without auto injection.
        """
        node = StudioLightSceneContract()
        custom_text = "portrait of an elegant woman, clean porcelain skin, high fashion editorial"
        contract, p1, p2, report = node.build(
            master_prompt="portrait of a model",
            environment="Auto",
            ambient_intent="Auto",
            refinement_mode="Custom Override",
            custom_pass2_prompt=f"  {custom_text}  ",
        )
        self.assertEqual(p2, custom_text)
        self.assertNotIn("preserving subject identity", p2)
        self.assertEqual(contract["refinement_mode"], "Custom Override")
        self.assertEqual(contract["effective_pass2_prompt"], custom_text)

        rep = json.loads(report)
        self.assertEqual(rep["refinement_mode"], "Custom Override")
        self.assertEqual(rep["effective_pass2_prompt"], custom_text)

    def test_scene_contract_custom_override_blank_raises_error(self):
        """
        Custom Override with empty text must raise ValueError (no silent fallback to Auto).
        """
        node = StudioLightSceneContract()
        with self.assertRaises(ValueError):
            node.build(
                master_prompt="portrait of a model",
                environment="Auto",
                ambient_intent="Auto",
                refinement_mode="Custom Override",
                custom_pass2_prompt="   ",
            )

    def test_scene_contract_legacy_invocation_backward_compatible(self):
        """
        Calling build() with only the 3 legacy arguments must default to Auto without error.
        """
        node = StudioLightSceneContract()
        contract, p1, p2, report = node.build(
            master_prompt="portrait of a model",
            environment="Auto",
            ambient_intent="Auto",
        )
        self.assertEqual(contract["refinement_mode"], "Auto")
        self.assertEqual(contract["effective_pass2_prompt"], p2)


if __name__ == "__main__":
    unittest.main()
