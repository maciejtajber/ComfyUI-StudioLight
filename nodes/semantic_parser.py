"""
ComfyUI-StudioLight — Semantic Scene Parser & Lighting Policy Resolver (v2.0)
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Implements the StudioLight Prompt & Albedo Architecture v2 specification:
- Decomposes master prompts into structured SceneDescription data models.
- Preserves 100% of physical scene identity, subject appearance, emotional markers (tears),
  clothing, wet surfaces, props, and material properties.
- Resolves conflicting lighting directives using two explicit policies:
  * RIG_OVERRIDE (default): StudioLight 3D rig is the absolute authority for directional
    illumination; original lighting directives are stripped from Pass 1 and Pass 2 prompts;
    physical emitters remain as objects without artificial illumination; no 80-word narrative concatenation.
  * HYBRID_PRACTICAL: Rig controls directional keys while in-scene practical emitters (screens,
    candles, neon) retain their intended emissive contributions.
- Eliminates fictional 'Log-C' buzzwords in favor of photographically grounded Neutral Diffuse Reference Capture.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class MaterialCategory(str, Enum):
    DIFFUSE = "DIFFUSE"
    DIELECTRIC_GLOSSY = "DIELECTRIC_GLOSSY"
    METALLIC = "METALLIC"
    MIRROR = "MIRROR"
    TRANSPARENT = "TRANSPARENT"
    TRANSLUCENT = "TRANSLUCENT"
    EMISSIVE = "EMISSIVE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class LightingPolicy(str, Enum):
    RIG_OVERRIDE = "RIG_OVERRIDE"
    HYBRID_PRACTICAL = "HYBRID_PRACTICAL"
    VERBATIM_RAW = "VERBATIM_RAW"


# --- PHOTOGRAPHIC GUIDANCE STRINGS ---

PASS1_NEUTRAL_DIFFUSE_CAPTURE = (
    "neutral diffuse reference capture, broad balanced illumination, "
    "preserved surface colors and textures, shadowless base exposure, "
    "recoverable detail in dark regions, no strong color cast"
)

PASS1_BALANCED_STUDIO_CAPTURE = (
    "broad balanced studio reference illumination, soft shadowless base exposure, "
    "clean surface textures and natural albedo"
)

PASS1_PRESERVE_MOOD_CAPTURE = (
    "soft even ambient illumination, preserved environmental base tone, "
    "balanced shadow depth with clear surface detail"
)

PASS2_MINIMAL_REFINEMENT = (
    "preserving subject identity, composition, material characteristics, "
    "and fine surface details"
)


# --- MATERIAL DICTIONARIES ---

MATERIAL_PATTERNS = [
    (
        MaterialCategory.METALLIC,
        re.compile(
            r"\b(liquid\s+chrome|chrome|gold|golden|silver|platinum|brass|copper|bronze|brushed\s+steel|titanium|metallic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MaterialCategory.DIELECTRIC_GLOSSY,
        re.compile(
            r"\b(patent\s+leather|high[- ]?gloss|glossy|gloss|lacquered|varnished|polished\s+marble|polished\s+stone|glazed|porcelain|latex|vinyl|shiny\s+leather)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MaterialCategory.TRANSPARENT,
        re.compile(
            r"\b(clear\s+glass|glass|crystal|transparent|clear\s+acrylic|water\s+droplets?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MaterialCategory.TRANSLUCENT,
        re.compile(
            r"\b(translucent|jade|frosted\s+glass|milk\s+glass|amber|alabaster|subsurface\s+scattering)\b",
            re.IGNORECASE,
        ),
    ),
    (
        MaterialCategory.DIFFUSE,
        re.compile(
            r"\b(cotton|wool|woolen|cashmere|tweed|felt|linen|canvas|denim|matte|chalk|stone|concrete)\b",
            re.IGNORECASE,
        ),
    ),
]


# --- PRACTICAL EMITTER CLASSIFICATION & PROP PRESERVATION ---

PRACTICAL_EMITTER_REWRITES = [
    # 0. Natural possessive action: "her face softly lit by ... a phone she holds in her hand"
    (
        re.compile(
            r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+(?:(?:is|are)\s+)?)?(?:(?:softly|gently|brightly|dimly)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?\s+(?:she|he|they)\s+holds?\s+in\s+(?P<holder>her|his|their)\s+(?P<hands>hands?)\b",
            re.IGNORECASE,
        ),
        r", holding a smartphone in \g<holder> \g<hands>",
        r", holding a smartphone with a cool glowing screen in \g<holder> \g<hands>",
        "smartphone",
    ),
    # 1. Screen glare with held action: e.g. "illuminated by the cool light of a smartphone screen held in her hands" -> "holding a smartphone in her hands"
    (
        re.compile(
            r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+(?:(?:is|are)\s+)?)?(?:(?:softly|gently|brightly|dimly)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?(?:\s+held\s+in\s+(?P<holder>her|his|their)\s+(?P<hands>hands?))?\b",
            re.IGNORECASE,
        ),
        r", holding a smartphone in \g<holder> \g<hands>",
        r", holding a smartphone with a cool glowing screen in \g<holder> \g<hands>",
        "smartphone",
    ),
    # 2. Screen glare without explicit held phrase: e.g. "illuminated by the cool light of a smartphone screen" -> ", with a smartphone"
    (
        re.compile(
            r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+(?:(?:is|are)\s+)?)?(?:(?:softly|gently|brightly|dimly)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?\b",
            re.IGNORECASE,
        ),
        ", with a smartphone",
        ", with a smartphone featuring a glowing screen",
        "smartphone",
    ),
    # 3. Direct participial phrase: "holding a smartphone illuminating her face" -> "holding a smartphone"
    (
        re.compile(
            r"\b((?:holding|with|carrying|using)\s+(?:a\s+|an\s+|the\s+)?(?:smartphone|phone)(?:\s+screen)?(?:\s+held\s+in\s+(?:her|his|their)\s+hands?)?)\s+"
            r"(?:illuminating|lighting|casting\s+light\s+on|bathing)\s+(?:her|his|their|the)?\s*(?:face|cheeks?|eyes?|features?|subject|room)\b",
            re.IGNORECASE,
        ),
        r"\1",
        r"\1 with glowing screen",
        "smartphone",
    ),
    # 4. Fire emitters: "illuminated by the light of a torch/candle/lantern held in his hand" -> ", holding a flaming torch in hand"
    (
        re.compile(
            r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+(?:(?:is|are)\s+)?)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+|golden\s+)?(?:light|glow|glare|illumination|flames?)\s+of\s+(?:a\s+)?(?P<item>torch|lantern|candle|candlestick|campfire)(?:\s+held\s+in\s+(?:her|his|their)\s+hands?)?\b",
            re.IGNORECASE,
        ),
        r", holding a \g<item> in hand",
        r", holding a \g<item> with warm glowing flame in hand",
        "fire_emitter",
    ),
    # 5. Neon emitters: "illuminated by neon lights" -> ", with glowing neon lights"
    (
        re.compile(
            r"(?:,\s*)?\b(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|red\s+|blue\s+|warm\s+)?neon\s+(?:lights?|glow|signs?)\b",
            re.IGNORECASE,
        ),
        ", with a neon sign",
        ", with a glowing neon sign",
        "neon_emitter",
    ),
]


# --- DIRECTIONAL LIGHTING DIRECTIVES TO NEUTRALIZE UNDER RIG_OVERRIDE ---

_SHADOW_ADJECTIVES = r"(?:crisp|sharp|deep|dark|stark|hard|harsh|heavy|black|long|angular|ambient|moody|soft|high[- ]?contrast)"
_LIGHT_ADJECTIVES = r"(?:harsh|hard|direct|blinding|overhead|frontal|on[- ]?camera|paparazzi|ring|strobe|xenon|bare[- ]?bulb|speedlight|dramatic|subtle|red|blue|warm|cool)"

DIRECTIONAL_LIGHTING_PATTERNS = [
    # 1. On-camera flash, strobe, speedlight with any leading verbs
    re.compile(
        rf"(?:,\s*)?\b(?:(?:and\s+|with\s+|under\s+|lit\s+by\s+|bathed\s+in\s+|illuminated\s+by\s+)?(?:a\s+|an\s+|the\s+)?)?"
        rf"(?:{_LIGHT_ADJECTIVES}\s+){{1,3}}"
        r"(?:flash(?:\s+photography)?|strobe|speedlight|bare[- ]?bulb|lighting|light)\b",
        re.IGNORECASE,
    ),
    # 2. Generic "illuminated by / lit by / bathed in [descriptors] [lighting / beams / rays / etc.]"
    re.compile(
        r"(?:,\s*)?\b(?:illuminated|lit|bathed)\s+by\s+(?:the\s+|a\s+|an\s+)?"
        r"(?:[a-zA-Z-]+\s+){0,4}"
        r"(?:light(?:ing)?|glow|glare|flash(?: photography)?|strobe|beams?|rays?|sunlight|illumination|rim|spotlight)\b"
        r"(?:\s+from\s+(?:the\s+)?(?:behind|above|below|left|right|front)(?:\s+side)?)?",
        re.IGNORECASE,
    ),
    # 3. "under [descriptors] [spotlight / sunlight / strobe / flash / etc.]"
    re.compile(
        r"(?:,\s*)?\bunder\s+(?:the\s+|a\s+|an\s+)?"
        r"(?:[a-zA-Z-]+\s+){0,3}"
        r"(?:spot[- ]?light|searchlight|floodlight|strobe|flash|sunlight|direct\s+light|lighting)\s*"
        r"(?:illumination|lighting|glare)?\b",
        re.IGNORECASE,
    ),
    # 4. Directional lights: rim light, kicker, backlight, side light, split lighting, sunlight
    re.compile(
        rf"(?:,\s*)?\b(?:(?:and\s+|with\s+|under\s+|in\s+)?(?:a\s+|an\s+|the\s+)?)?"
        rf"(?:{_LIGHT_ADJECTIVES}\s+){{0,3}}"
        r"(?:rim[- ]?light(?:ing)?|kicker(?:\s+light)?|back[- ]?light(?:ing)?|side[- ]?light(?:ing)?|split[- ]?lighting|sunlight|sunbeams?|sun\s+rays?|midday\s+sun|direct\s+sun|key[- ]?light(?:ing)?|spot[- ]?light(?:ing)?)\b"
        r"(?:\s+from\s+(?:the\s+)?(?:behind|above|below|left|right|front)(?:\s+side)?)?",
        re.IGNORECASE,
    ),
    # 5. Casting shadows clauses: e.g. "deep dark shadows casting on the concrete wall"
    re.compile(
        rf"(?:,\s*)?(?<!\beye\s)(?<!\bo'clock\s)(?<!\bbeard\s)\b(?:and\s+|with\s+)?"
        rf"(?:{_SHADOW_ADJECTIVES}\s+){{0,4}}shadows\s+(?:casting|falling|projected)"
        r"(?:\s+(?:on|onto|across|against|over|behind)\s+[a-zA-Z-]+(?:\s+[a-zA-Z-]+){0,3})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:,\s*)?(?<!\beye\s)(?<!\bo'clock\s)(?<!\bbeard\s)\b(?:and\s+|with\s+)?(?:casting|casts)\s+(?:a\s+|an\s+|the\s+)?"
        rf"(?:{_SHADOW_ADJECTIVES}\s+){{0,4}}shadows?"
        r"(?:\s+(?:on|onto|across|against|over|behind)\s+[a-zA-Z-]+(?:\s+[a-zA-Z-]+){0,3})?\b",
        re.IGNORECASE,
    ),
    # Compound shadow adjectives: e.g. "crisp sharp shadows", "deep dark shadows"
    re.compile(
        rf"(?:,\s*)?(?<!\beye\s)(?<!\bo'clock\s)(?<!\bbeard\s)\b(?:and\s+|with\s+)?"
        rf"(?:{_SHADOW_ADJECTIVES}\s+){{1,4}}shadows?\b",
        re.IGNORECASE,
    ),
    # 6. In darkness clauses: "sitting in total darkness", "in darkness"
    re.compile(
        r"(?:,\s*)?\b(?:sitting\s+|standing\s+|lying\s+)?(?:in|into|amid|within)\s+"
        r"(?:the\s+)?"
        r"(?:pitch[- ]?blackness|pitch[- ]?darkness|total\s+darkness|complete\s+darkness|utter\s+darkness|absolute\s+darkness|darkness)\b",
        re.IGNORECASE,
    ),
    # 7. Chiaroscuro & high contrast
    re.compile(
        r"(?:,\s*)?\b(?:dramatic|heavy|stark|deep|intense|extreme|high[- ]?contrast|classic)?\s*"
        r"(?:chiaroscuro(?:\s+lighting)?|tenebrism|caravaggio\s+lighting|rembrandt\s+lighting|split\s+lighting|"
        r"half[- ]?lit\s+face|half\s+in\s+shadow|film\s+noir\s+lighting|stark\s+contrast|harsh\s+contrast)\b",
        re.IGNORECASE,
    ),
]


@dataclass
class SceneDescription:
    """
    Structured intermediate representation of a scene prompt.
    Traceable, deterministic, and material-aware.
    """
    original_prompt: str
    scene_content: str
    material_descriptions: List[str] = field(default_factory=list)
    material_categories: List[str] = field(default_factory=list)
    environmental_context: str = "universal"
    practical_emitters: List[Dict[str, Any]] = field(default_factory=list)
    lighting_directives: List[str] = field(default_factory=list)
    unresolved_fragments: List[str] = field(default_factory=list)
    policy_applied: str = LightingPolicy.RIG_OVERRIDE.value
    pass1_prompt: str = ""
    pass2_prompt: str = ""
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "scene_content": self.scene_content,
            "material_descriptions": self.material_descriptions,
            "material_categories": self.material_categories,
            "environmental_context": self.environmental_context,
            "practical_emitters": self.practical_emitters,
            "lighting_directives": self.lighting_directives,
            "policy_applied": self.policy_applied,
            "pass1_prompt": self.pass1_prompt,
            "pass2_prompt": self.pass2_prompt,
        }


def parse_scene_description(
    prompt: str,
    lighting_policy: str = LightingPolicy.RIG_OVERRIDE.value,
) -> SceneDescription:
    """
    Parses an arbitrary master prompt into a structured SceneDescription.
    Extracts materials, practical emitters, and directional lighting directives
    without deleting subject identity, tears, makeup, poses, or clothing.
    """
    raw = prompt.strip()
    text = raw
    directives: List[str] = []
    emitters: List[Dict[str, Any]] = []
    mat_descs: List[str] = []
    mat_cats: Set[str] = set()

    # 1. Detect Materials
    for cat, pat in MATERIAL_PATTERNS:
        matches = pat.findall(raw)
        if matches:
            mat_cats.add(cat.value)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                if m.lower() not in [d.lower() for d in mat_descs]:
                    mat_descs.append(m.lower())

    # 2. Extract and preserve practical emitters
    is_hybrid = (lighting_policy == LightingPolicy.HYBRID_PRACTICAL.value)
    for pat, repl_override, repl_hybrid, em_name in PRACTICAL_EMITTER_REWRITES:
        repl = repl_hybrid if is_hybrid else repl_override
        for m in pat.finditer(text):
            directives.append(m.group(0).strip(" ,."))
            emitters.append({"type": em_name, "raw": m.group(0), "mode": "hybrid" if is_hybrid else "override"})
        text = pat.sub(repl, text)

    # 3. Strip directional lighting directives
    for pat in DIRECTIONAL_LIGHTING_PATTERNS:
        for m in pat.finditer(text):
            clause = m.group(0).strip(" ,.")
            if clause and clause not in directives:
                directives.append(clause)
        text = pat.sub("", text)

    # 4. Clean up grammar and punctuation
    text = re.sub(r"\bunderground\s+dark\s+(techno\s+club|club|bar)\b", r"underground \1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:and|with|in|under|by|from|casting)\s*(?=[,.]|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:^|(?<=,))\s*\b(?:and|with)\b\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s+([,.])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.")

    # 5. Environmental context
    env_match = re.search(
        r"\b(underground\s+club|nightclub|club|cavern|cave|night|sunset|sunrise|golden\s+hour|photo\s+studio|studio|interior|outdoor)\b",
        raw,
        re.I,
    )
    env_ctx = env_match.group(0).lower() if env_match else "universal"

    return SceneDescription(
        original_prompt=raw,
        scene_content=text,
        material_descriptions=mat_descs,
        material_categories=list(mat_cats),
        environmental_context=env_ctx,
        practical_emitters=emitters,
        lighting_directives=directives,
        policy_applied=lighting_policy,
    )


def build_pass1_prompt_v2(
    scene: SceneDescription,
    capture_guidance: str = "Neutral Diffuse Reference",
    albedo_suffix: str = "auto",
) -> str:
    """
    Constructs the Pass 1 prompt:
    Scene Content + Material Preservations + Neutral Diffuse Reference Capture.
    Guarantees shadowless, flat base albedo without corrupting clothing or materials.
    """
    text = scene.scene_content

    # Material preservation phrasing
    mat_preservation = ""
    if MaterialCategory.METALLIC.value in scene.material_categories and "chrome" in text.lower():
        if "reflective metallic" not in text.lower():
            mat_preservation = ", with reflective metallic surfaces"

    # Select capture guidance
    if capture_guidance == "Balanced Studio Capture":
        guidance = PASS1_BALANCED_STUDIO_CAPTURE
    elif capture_guidance == "Preserve Environmental Tone":
        guidance = PASS1_PRESERVE_MOOD_CAPTURE
    elif capture_guidance == "Legacy Log-C":
        guidance = (
            "Evenly lit flat logarithmic sensor capture, 360-degree omnidirectional soft diffused ambient illumination, "
            "uniform diffuse ambient light across subject and background"
        )
    else:  # Default: Neutral Diffuse Reference
        guidance = PASS1_NEUTRAL_DIFFUSE_CAPTURE

    # Optional custom suffix handling
    suffix_clause = ""
    if albedo_suffix and albedo_suffix.strip() not in ("", "auto"):
        clean_sfx = albedo_suffix.strip().strip(",")
        suffix_clause = f", {clean_sfx}"

    pass1 = f"{text}{mat_preservation}, {guidance}{suffix_clause}"
    pass1 = re.sub(r"\s*,\s*,+", ", ", pass1)
    pass1 = re.sub(r"\s+([,.])", r"\1", pass1).strip(" ,.")
    return pass1


def build_pass2_prompt_v2(
    scene: SceneDescription,
    lighting_policy: str = LightingPolicy.RIG_OVERRIDE.value,
    minimal_refinement: bool = True,
) -> str:
    """
    Constructs the Pass 2 prompt:
    Under RIG_OVERRIDE: Scene Content + Material Guidance + Minimal Refinement Tag.
    Strictly eliminates contradictory lighting directives and does NOT concatenate the 80-word narrative.
    Under HYBRID_PRACTICAL: Keeps practical emitter glow descriptions.
    Under VERBATIM_RAW: Retains raw master prompt (legacy v1 behavior).
    """
    if lighting_policy == LightingPolicy.VERBATIM_RAW.value:
        return scene.original_prompt.strip()

    text = scene.scene_content

    # Material preservation phrasing
    mat_preservation = ""
    if MaterialCategory.METALLIC.value in scene.material_categories and "chrome" in text.lower():
        if "reflective metallic" not in text.lower():
            mat_preservation = ", with reflective metallic surfaces"

    refinement_clause = f", {PASS2_MINIMAL_REFINEMENT}" if minimal_refinement else ""

    pass2 = f"{text}{mat_preservation}{refinement_clause}"
    pass2 = re.sub(r"\s*,\s*,+", ", ", pass2)
    pass2 = re.sub(r"\s+([,.])", r"\1", pass2).strip(" ,.")
    return pass2
