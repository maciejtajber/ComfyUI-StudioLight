"""
ComfyUI-StudioLight — High-Precision Universal Semantic Prompt Sanitizer (v2.1)
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Decouples zero-photon darkness, harsh cast shadows, and direct flash traps between
Pass 1 (Albedo Draft) and Pass 2 (Lit Synthesis) using a 5-Layer Universal Semantic
Chess Dictionary + Stage 2A Log-C Flat Acquisition framing. Preserves 100% of subject
identity, clothing, specular materials, practical emitters, and environmental time-of-day.
"""

import re
from typing import Tuple
from .semantic_parser import (
    parse_scene_description,
    build_pass1_prompt_v2,
    build_pass2_prompt_v2,
    LightingPolicy,
    PASS1_NEUTRAL_DIFFUSE_CAPTURE,
    PASS2_MINIMAL_REFINEMENT,
)


# ==============================================================================
# LAYER 0: PROTECTED SUBJECT, ANATOMY, CLOTHING & ACCESSORY IMMUNITY
# Never alter "dark", "black", "deep", or "shadow" when describing the person,
# their hair/eyes/skin/makeup, or their clothing, fabrics, and accessories.
# ==============================================================================
PROTECTED_SUBJECT_NOUNS = (
    r"hair|haired|locks|braids|curls|waves|bangs|fringe|ponytail|bun|chignon|"
    r"beard|bearded|mustache|stubble|brows|eyebrows|lashes|eyelashes|"
    r"eyes|eyed|iris|irises|pupils|gaze|"
    r"skin|skinned|complexion|tone|freckles|tan|"
    r"lips|lipstick|lipgloss|makeup|cosmetics|eyeshadow|eyeliner|mascara|blush|contour|nails|manicure|"
    r"coat|overcoat|trenchcoat|trench|peacoat|parka|jacket|blazer|windbreaker|"
    r"suit|tuxedo|tailoring|vest|waistcoat|"
    r"dress|gown|robe|kimono|sari|abaya|cloak|cape|poncho|shawl|"
    r"shirt|blouse|top|turtleneck|polo|tee|t-shirt|tank\s+top|corset|bodysuit|bustier|bra|lingerie|"
    r"sweater|jumper|pullover|cardigan|hoodie|sweatshirt|knitwear|"
    r"pants|trousers|slacks|chinos|jeans|denim|shorts|skirt|miniskirt|leggings|tights|stockings|hosiery|socks|"
    r"shoes|boots|heels|stilettos|pumps|sneakers|trainers|loafers|oxfords|sandals|slippers|footwear|"
    r"gloves|mittens|hat|cap|beret|fedora|beanie|helmet|hood|headscarf|hijab|turban|veil|"
    r"scarf|necktie|tie|bowtie|belt|suspenders|"
    r"bag|handbag|purse|clutch|tote|satchel|backpack|briefcase|luggage|wallet|"
    r"sunglasses|glasses|eyeglasses|spectacles|frames|goggles|visor|mask|"
    r"watch|wristwatch|jewelry|necklace|pendant|choker|earrings|bracelet|bangle|ring|brooch|"
    r"wool|woolen|cashmere|tweed|houndstooth|felt|"
    r"leather|suede|patent|nubuck|sheepskin|shearling|"
    r"velvet|velour|silk|satin|chiffon|organza|tulle|lace|mesh|netting|"
    r"cotton|linen|canvas|twill|gabardine|corduroy|flannel|fleece|jersey|"
    r"nylon|polyester|spandex|lycra|latex|vinyl|pvc|rubber|neoprene|acrylic|"
    r"fur|feathers|sequins|beads|embroidery|"
    r"fabric|textile|cloth|material|weave|drapery|upholstery|"
    r"clothing|clothes|garment|garments|attire|outfit|wardrobe|costume|uniform|armor|armour|"
    r"chocolate|coffee|espresso|wine|beer|stout|tea|liquor|"
    r"horse|stallion|dog|hound|wolf|panther|jaguar|cat|raven|crow"
)

# Matches 0 to 3 descriptive adjectives before a protected subject/clothing noun
# e.g. "dark swept-back hair", "dark-skinned", "dark structured Savile Row wool coat"
_PROTECTED_MODIFIER_CHAIN = rf"(?:[\s-]+(?!and\b|or\b|with\b|in\b|at\b|on\b|under\b|from\b|by\b)[a-zA-Z]+){{0,3}}[\s-]+\b(?:{PROTECTED_SUBJECT_NOUNS})\b"


# ==============================================================================
# LAYER 1: UNIVERSAL DARKNESS / ZERO-PHOTON BLINDNESS DICTIONARY (PASS 1 ONLY)
# Zero hardcoded location nouns. Matches darkness/blindness modifiers universally
# across the English language whenever not modifying a protected subject/clothing noun.
# ==============================================================================
STANDALONE_DARKNESS_IDIOMS = [
    # Specific indoor/underground club or cavern darkness -> evenly illuminated low-key with clear surface detail
    (
        r"\bdark\s+(?:underground|club|nightclub|bar|rave)\b",
        "evenly illuminated underground club with clear surface detail",
    ),
    (
        r"\b(?:dark|pitch[- ]?black)\s+(?:stone\s+)?(?:cave|cavern)\b",
        "evenly illuminated low-key cavern with clearly visible rock texture",
    ),
    # Pitch-blackness / total darkness noun phrases (or "in the dark" followed by clause boundary)
    (
        r"\b(?:in|into|inside|within|through|amid|amidst)\s+"
        r"(?:the\s+|a\s+)?"
        r"(?:pitch[- ]?blackness|pitch[- ]?darkness|total\s+darkness|complete\s+darkness|"
        r"utter\s+darkness|absolute\s+darkness|dead\s+of\s+night|inky\s+darkness|velvet\s+darkness|"
        r"stygian\s+darkness|abyssal\s+darkness|deep\s+darkness|darkness|"
        r"(?:pitch[- ]?black|pitch[- ]?dark|dark)(?=\s*(?:[,.]|and\b|or\b|with\b|where\b|while\b|$)))",
        "in soft even low-key ambient illumination",
    ),
    # Shrouded / cloaked / swallowed by darkness or shadow
    (
        r"\b(?:shrouded|cloaked|swallowed|engulfed|bathed|plunged|submerged|immersed|enveloped|"
        r"wrapped|veiled|draped|hidden|concealed|obscured|lost|emerging|stepping)\s+"
        r"(?:in|into|by|from|out\s+of)\s+(?:the\s+|a\s+)?"
        r"(?:deep\s+|heavy\s+|total\s+|pitch\s+|inky\s+|thick\s+)?"
        r"(?:darkness|dark|shadows?|shade|gloom|murk)\b",
        "in soft even low-key ambient illumination",
    ),
    # In the shadows / in deep shadow
    (
        r"\b(?:in|into|within|under|beneath|inside|from)\s+(?:the\s+)?"
        r"(?:deep\s+|heavy\s+|dark\s+|thick\s+|dense\s+|cast\s+|looming\s+|ambient\s+)?"
        r"shadows?\b",
        "in soft low-key ambient light",
    ),
    # Zero light / absence of light / starved of light
    (
        r"\b(?:with\s+|under\s+|in\s+)?"
        r"(?:zero\s+natural\s+daylight|zero\s+natural\s+light|no\s+natural\s+daylight|no\s+natural\s+light|"
        r"no\s+ambient\s+light|zero\s+ambient\s+light|zero\s+light|no\s+light\s+source|no\s+light|"
        r"absence\s+of\s+light|devoid\s+of\s+light|starved\s+of\s+light|without\s+any\s+light)\b",
        "with soft low-key ambient fill",
    ),
    # Visibility / underexposure killers
    (
        r"\b(?:barely|scarcely|hardly|faintly)\s+(?:visible|lit|illuminated|discernible)\b",
        "clearly discernible in soft low-key light",
    ),
    (
        r"\b(?:severely\s+|heavily\s+|deeply\s+)?underexposed\b",
        "balanced low-key exposure",
    ),
    (
        r"\b(?:crushed|clipped|blocked[- ]?up|inky|opaque)\s+(?:blacks?|shadows?)\b",
        "preserved low-key shadow detail",
    ),
]

# Universal Extreme Zero-Photon Adjectives -> "evenly exposed low-key" (when not followed by protected subject/clothing)
EXTREME_DARKNESS_ADJECTIVES = (
    r"\b(?:pitch[- ]?black|pitch[- ]?dark|ink[- ]?black|inky[- ]?black|coal[- ]?black|"
    r"stygian|abyssal|lightless|sunless|moonless|starless|unlit|unilluminated|"
    r"blacked[- ]?out|windowless\s+dark|completely\s+dark|totally\s+dark|utterly\s+dark|"
    r"dead\s+dark|deeply\s+dark)"
    rf"(?!{_PROTECTED_MODIFIER_CHAIN})\b"
)

# Universal Standard Darkness Adjectives -> "dusky low-key" (when not followed by protected subject/clothing)
# Also excludes "dark blue/green/red/amber/gold/brown/grey/crimson/emerald/violet/purple/cyan/teal/magenta" color names
COLOR_HUE_WORDS = (
    r"blue|navy|indigo|azure|cobalt|cyan|teal|turquoise|"
    r"green|emerald|olive|jade|forest|moss|lime|"
    r"red|crimson|scarlet|burgundy|maroon|wine|ruby|"
    r"purple|violet|plum|magenta|lavender|lilac|"
    r"brown|mahogany|chestnut|umber|sepia|tan|beige|ochre|"
    r"grey|gray|charcoal|slate|silver|pewter|"
    r"orange|amber|gold|golden|yellow|bronze|copper"
)

STANDARD_DARKNESS_ADJECTIVES = (
    r"\b(?:dark|gloomy|murky|tenebrous|shadowy|shaded|dingy|dismal|dreary|"
    r"dimly[- ]?lit|poorly[- ]?lit|low[- ]?lit|half[- ]?dark|semi[- ]?dark)"
    rf"(?![\s-]+\b(?:{COLOR_HUE_WORDS})\b)"
    rf"(?!{_PROTECTED_MODIFIER_CHAIN})\b"
)


# ==============================================================================
PRACTICAL_EMITTER_AND_PROP_REWRITES = [
    # Exact beta phrasing: "her face softly lit by the cool glow of a smartphone she holds in her hand".
    # Remove the screen-emission instruction as one complete clause so later lighting cleanup cannot
    # leave fragments such as "her face softly, a smartphone she holds in her hand".
    (
        r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+)?"
        r"(?:softly|gently|brightly|dimly)\s+lit\s+by\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?"
        r"(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?"
        r"\s+she\s+holds?\s+in\s+her\s+hand(?:s)?\b",
        ", holding a smartphone in her hand",
    ),
    # "face illuminated by the cool light of a smartphone screen held in her hands" -> "holding a smartphone in her hands"
    (
        r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?(?:\s+held\s+in\s+(?:her|his|their)\s+hands)\b",
        ", holding a smartphone in her hands",
    ),
    # "illuminated by the cool light of a smartphone screen" -> ", with a smartphone"
    (
        r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination)\s+of\s+(?:a\s+)?(?:smartphone|phone)(?:\s+screen)?\b",
        ", with a smartphone",
    ),
    # "face illuminated by the light of a torch held in his hand" -> ", holding a flaming torch in hand"
    (
        r"(?:,\s*)?\b(?:(?:her|his|their|the|a)?\s*(?:face|cheeks?|eyes?|features?)\s+)?(?:illuminated\s+by|lit\s+by|bathed\s+in)\s+(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?(?:light|glow|glare|illumination|flames?)\s+of\s+(?:a\s+)?(?:torch|lantern|candle|campfire)(?:\s+held\s+in\s+(?:her|his|their)\s+hand(?:s)?)\b",
        ", holding a flaming torch in hand",
    ),
]


BACKLIT_SILHOUETTE_REWRITES = [
    (
        r"\b(?:back[- ]?lit\s+silhouette|dark\s+silhouette|black\s+silhouette|stark\s+silhouette|"
        r"in\s+silhouette|silhouetted\s+against|silhouette(?:d)?|contre[- ]?jour|"
        r"obscured\s+face\s+in\s+backlight)\b",
        "softly front-filled subject",
    ),
]

_SHADOW_ADJ = (
    r"(?:long|hard|harsh|sharp|razor[- ]?sharp|crisp|deep|dark|black|heavy|stark|dramatic|"
    r"ambient|moody|soft|side|dense|tenebrous|facial|eye[- ]?socket|"
    r"elongated|diagonal|slanted|angular|ominous|looming|menacing|distinct|clear|projected|"
    r"cast|wall|floor|background|drop|double|crossed|silhouette|high[- ]?contrast|"
    r"dappled|mottled|venetian\s+blind|window\s+frame|window\s+blind|blinds|gobo|lace|foliage|tree\s+branch|barred|cage|slatted|striped)"
)

_LIGHT_ADJ = (
    r"(?:harsh|hard|direct|blinding|piercing|searing|scorching|glaring|intense|extreme|stark|"
    r"overhead|frontal|on[- ]?camera|paparazzi|ring|pop[- ]?up|strobe|xenon|bare[- ]?bulb|cobra|"
    r"studio|single|strong|dramatic|warm|cool|blue|cyan|amber|golden|neon|cinematic|directional|diagonal|slanted|angul(?:ar|ed)|specular)"
)

HARSH_SHADOW_AND_FLASH_PATTERNS = [
    # 0. Practical Emitters, Screen Glare & Tear Traps (Pass 1 Albedo decoupling - matched first before generic light adjectives)
    r"(?:,\s*)?\b(?:(?:and|with|under|illuminated\s+by|lit\s+by|bathed\s+in)\s+)?(?:the\s+)?(?:cool\s+|blue\s+|warm\s+)?light\s+of\s+(?:a\s+)?(?:smartphone|phone|screen|monitor|torch|lantern)\b",
    r"(?:,\s*)?\b(?:(?:and|with|under|illuminated\s+by|lit\s+by)\s+)?(?:the\s+|a\s+)?(?:smartphone|phone|screen|monitor)\s+(?:glow|glare|illumination|light)\b",
    r"(?:,\s*)?\b(?:illuminated|lit)\s+by\s+(?:a\s+|the\s+)?(?:smartphone|phone|screen|monitor)\b",
    r"(?:,\s*)?\b(?:and\s+|with\s+)?(?:the\s+)?tears\s+(?:are\s+)?(?:accumulating|pooling|streaming|flowing|running|welling)\s+(?:in\s+|down\s+|on\s+)(?:her\s+|his\s+|their\s+)?(?:eyes?|cheeks?|face)\b",
    r"(?:,\s*)?\b(?:and\s+|with\s+)?(?:crying|weeping|teardrops?)\b",
    r"(?:,\s*)?\b(?:and\s+|with\s+)?tears\b",
    r"(?:,\s*)?\b(?:and\s+|with\s+)?wet\s+(?:cheeks?|face|skin)\b",

    # 1. Complete "casting [adj] shadows on [surface]" clauses
    rf"(?:,\s*)?\b(?:and\s+|with\s+)?(?:casting|casts)\s+(?:a\s+|an\s+|the\s+)?(?:{_SHADOW_ADJ}\s+){{0,4}}shadows?"
    r"(?:\s+(?:on|onto|across|against|over|behind|along)\s+(?:the\s+|a\s+|her\s+|his\s+|their\s+)?[a-zA-Z-]+(?:\s+[a-zA-Z-]+){0,3})?\b",

    # 2. Any 1 to 4 shadow adjectives + "shadow(s)" (excludes "eye shadow" / "o'clock shadow" / "beard shadow")
    rf"(?:,\s*)?(?<!\beye\s)(?<!\bo'clock\s)(?<!\bbeard\s)\b(?:and\s+|with\s+)?(?:{_SHADOW_ADJ}\s+){{1,4}}shadows?"
    r"(?:\s+(?:on|onto|across|against|over|behind|along)\s+(?:the\s+|a\s+|her\s+|his\s+|their\s+)?[a-zA-Z-]+(?:\s+[a-zA-Z-]+){0,3})?\b",

    # 3. Chiaroscuro / Tenebrism / Split / Half-Shadowed Face Traps (supports leading adjectives like "dramatic chiaroscuro")
    r"(?:,\s*)?\b(?:and\s+|with\s+)?(?:(?:dramatic|heavy|stark|deep|intense|extreme|high[- ]?contrast|classic|baroque|moody)\s+)?"
    r"(?:chiaroscuro(?:\s+lighting|\s+shadows?)?|tenebrism|caravaggio\s+lighting|rembrandt\s+lighting|split\s+lighting|"
    r"half[- ]?lit\s+face|half\s+in\s+shadow|half[- ]?shadowed\s+face|film\s+noir\s+lighting|noir\s+shadows?|"
    r"stark\s+contrast|extreme\s+contrast|brutal\s+contrast|harsh\s+contrast|high[- ]?contrast\s+shadows?)\b",

    # 4. On-Camera Flash, Strobe, & Blinding Directional Point-Sources (supports 1 to 3 adjectives like "harsh overhead spotlight")
    rf"(?:,\s*)?\b(?:and\s+|with\s+|under\s+|lit\s+by\s+|bathed\s+in\s+)?(?:a\s+|an\s+|the\s+)?(?:{_LIGHT_ADJ}\s+){{1,3}}"
    r"(?:flash(?:\s+photography)?|speedlight|strobe|spot[- ]?light(?:ing)?|searchlight|floodlight|headlight\s+glare|"
    r"beam|rays?|light\s+shafts?|god\s+rays?|glare|specular\s+glare|sunlight|midday\s+sun|sunbeams?|"
    r"rim[- ]?light(?:ing)?|back[- ]?light(?:ing)?|side[- ]?light(?:ing)?|key[- ]?light(?:ing)?|uplight(?:ing)?|downlight(?:ing)?|light(?:ing)?)\b"
    r"(?:\s+(?:from|across|through|onto|in)\s+(?:the\s+)?(?:left|right|behind|above|below|side|rear|front|wall|floor|room|skylight|window|background))?",

    # 4b. Bare directional light nouns (e.g. "hard sunlight", "rim light from the left", "god rays")
    r"(?:,\s*)?\b(?:and\s+|with\s+|under\s+|lit\s+by\s+|bathed\s+in\s+)?(?:a\s+|an\s+|the\s+)?"
    r"(?:rim[- ]?light(?:ing)?|back[- ]?light(?:ing)?|back[- ]?lit|sunlight|sunbeams?|sun\s+rays?|"
    r"light\s+shafts?|god\s+rays?|volumetric\s+(?:god\s+)?rays?|key[- ]?light(?:ing)?|side[- ]?light(?:ing)?|"
    r"spot[- ]?light(?:ing)?|spotlight|hair[- ]?light(?:ing)?|kicker(?:\s+light)?)\b"
    r"(?:\s+(?:from|across|through|onto|in)\s+(?:the\s+)?(?:left|right|behind|above|below|side|rear|front|wall|floor|room|skylight|window|background))?",

    # 5. Overexposure / Clipping Traps
    r"(?:,\s*)?\b(?:and\s+|with\s+)?(?:blown[- ]?out|clipped|overexposed|searing|nuclear|blinding\s+white)\s+(?:highlights?|glare|whites?|speculars?)\b",
]


# ==============================================================================
# LAYER 4: SCENE-ADAPTIVE PASS 1 SUFFIXES (NO T5-POISONING 'CAST SHADOWS' TOKENS)
# ==============================================================================
PHOTOMETRIC_SUFFIXES = {
    "low_key": (
        ", diffuse low-key ambient illumination, soft shadowless ambient fill on subject, "
        "clean natural surface albedo, balanced shadow depth"
    ),
    "warm_ambient": (
        ", soft diffused warm ambient fill, even shadowless base exposure, "
        "clean natural surface albedo"
    ),
    "studio_explicit": (
        ", diffuse even studio ambient illumination, soft shadowless base plate across center and corners, "
        "no backdrop vignette, clean natural surface albedo"
    ),
    "universal": (
        ", diffuse even ambient illumination, soft shadowless base plate, "
        "clean natural surface albedo"
    ),
}

# Legacy alias for backward compatibility
SCENE_ALBEDO_SUFFIXES = PHOTOMETRIC_SUFFIXES

LEGACY_DEFAULT_SUFFIXES = {
    "",
    "auto",
    ", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog",
    ", flat overcast diffuse daylight, soft even shadowless lighting, rich natural color, beauty catalog",
    ", flat diffuse high-key ambient lighting, shadowless, Vogue cover style",
}


def detect_photometric_exposure(prompt: str) -> str:
    """
    Detects only the ambient exposure state ('studio_explicit', 'low_key', 'warm_ambient', or 'universal').
    Never guesses physical locations or injects studio backdrops into non-studio prompts.
    """
    p = prompt.lower()

    # 1. Explicit Studio ONLY when the user explicitly asks for a studio/cyclorama/backdrop
    studio_explicit_kw = (
        r"\b(photo\s+studio|studio\s+portrait|studio\s+shoot|studio\s+backdrop|studio\s+background|"
        r"cyclorama|cyc\s+wall|seamless\s+paper|seamless\s+backdrop|grey\s+backdrop|white\s+backdrop|"
        r"black\s+backdrop)\b"
    )
    if re.search(studio_explicit_kw, p):
        return "studio_explicit"

    # 2. Low-Key / Night / Dusky / Dark Moods (preserves dark ambient baseline without washing into daylight)
    low_key_kw = (
        r"\b(night|nighttime|midnight|late\s+night|noir|cyberpunk|dark|darkness|dusky|dim|dimly|"
        r"gloomy|murky|tenebrous|shadowy|unlit|lightless|pitch[- ]?black|pitch[- ]?dark|"
        r"candle(?:light|lit)?|torch(?:light|lit)?|flaming\s+torch|burning\s+torch|lantern|"
        r"campfire|bonfire|moon(?:light|lit)?|nocturnal|dusk|twilight|neon|underground|club|nightclub|rave|bar|pub)\b"
    )
    if re.search(low_key_kw, p):
        return "low_key"

    # 3. Warm / Golden Hour Moods
    warm_kw = r"\b(sunset|sunrise|golden\s+hour|magic\s+hour|amber\s+sky|dusk\s+glow)\b"
    if re.search(warm_kw, p):
        return "warm_ambient"

    # 4. Universal Baseline
    return "universal"


def detect_scene_context(prompt: str) -> str:
    """
    Backward-compatible alias for detect_photometric_exposure.
    Maps exposure modes to legacy keys if requested by existing scripts.
    """
    mode = detect_photometric_exposure(prompt)
    legacy_map = {
        "studio_explicit": "studio",
        "low_key": "night",
        "warm_ambient": "golden_hour",
        "universal": "universal",
    }
    return legacy_map.get(mode, mode)


def sanitize_pass1_prompt_universal(
    master_prompt: str,
    albedo_suffix: str = "auto",
    pass1_log_acquisition: bool = True,
) -> str:
    """
    Applies the 5-Layer Universal Semantic Chess Dictionary to generate the Pass 1
    Albedo prompt while keeping 100% of subject, clothing, props, and atmosphere intact.
    When pass1_log_acquisition=True, wraps Pass 1 in Stage 2A Log-C Flat Acquisition
    framing (preserving specular material nouns while guaranteeing shadowless facial exposure).
    """
    cleaned = master_prompt

    # Step 1: Translate standalone zero-photon / darkness idioms
    for pat, repl in STANDALONE_DARKNESS_IDIOMS:
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)

    # Step 1.5: Translate practical emitter & screen traps into natural props (prevents orphan "face screen" nouns)
    for pat, repl in PRACTICAL_EMITTER_AND_PROP_REWRITES:
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)

    # Step 2: Translate silhouette / contre-jour traps into front-filled subject
    for pat, repl in BACKLIT_SILHOUETTE_REWRITES:
        cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)

    # Step 3: Universal Extreme & Standard Darkness Adjectives -> "evenly exposed low-key" / "dusky low-key"
    # (Automatically protects hair, eyes, skin, clothing, fabrics, and color hues via negative lookahead)
    cleaned = re.sub(EXTREME_DARKNESS_ADJECTIVES, "evenly exposed low-key", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(STANDARD_DARKNESS_ADJECTIVES, "dusky low-key", cleaned, flags=re.IGNORECASE)

    # Collapse accidental double "dusky low-key dusky low-key" if two dark adjectives were adjacent
    cleaned = re.sub(r"\b(dusky\s+low-key)(?:\s+\1)+\b", r"\1", cleaned, flags=re.IGNORECASE)

    # Step 4: Strip harsh cast shadows, chiaroscuro, direct flash, and blinding glare traps
    for pat in HARSH_SHADOW_AND_FLASH_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    if not pass1_log_acquisition:
        # Legacy mode: decouple high-specularity tokens in Pass 1
        cleaned = re.sub(r"\b(overhead\s+)?skylights?\b", "diffused ceiling panels", cleaned, flags=re.IGNORECASE)
        legacy_material_rewrites = [
            (r"\bpatent\s+leather\b", "smooth leather"),
            (r"\b(?:liquid\s+)?chrome\b", "matte brushed metal"),
        ]
        for pat, repl in legacy_material_rewrites:
            cleaned = re.sub(pat, repl, cleaned, flags=re.IGNORECASE)

        specular_tokens = [
            r"\bultra[- ]?glossy\b", r"\bhigh[- ]?gloss\b", r"\bglossy\b",
            r"\bpatent\b", r"\bshiny\b", r"\blacquered\b", r"\bvarnished\b",
            r"\breflective\b", r"\bspecular\b", r"\bglare\b", r"\bsunlit\b",
        ]
        for token in specular_tokens:
            cleaned = re.sub(token, "", cleaned, flags=re.IGNORECASE)

    # Step 5: Clean up dangling prepositions, conjunctions, articles, and punctuation
    cleaned = re.sub(r"\b(by|with|from|in|under|on|onto)\s+of\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(by|with|from|in|under|on|onto|of)\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\ban\s+(?=dusky\b)", "a ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\ba\s+(?=evenly\b)", "an ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:and|with|in|under|by|from|casting)\s*(?=[,.]|$)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?:^|(?<=,))\s*\b(?:and|with)\b\s*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.")

    if pass1_log_acquisition:
        # Stage 2A Log-C Flat Capture Framing (Proven on 00053_ Techno Girl):
        # Forces FLUX Pass 1 to render shadowless, evenly lit faces with full surface detail
        # while keeping dark/night clubs, caves, and outdoor night skies faithful.
        prefix = (
            "Evenly lit flat logarithmic sensor capture, 360-degree omnidirectional soft diffused ambient illumination, "
            "uniform diffuse ambient light across subject and background: "
        )
        raw_suffix = (albedo_suffix or "").strip()
        if raw_suffix in LEGACY_DEFAULT_SUFFIXES or raw_suffix.lower() == "auto":
            exposure_mode = detect_photometric_exposure(master_prompt)
            mode_tail = PHOTOMETRIC_SUFFIXES[exposure_mode]
            suffix = (
                f"{mode_tail}, soft omnidirectional diffuse ambient illumination, "
                "uniform even exposure across subject and surfaces, clear natural surface texture"
            )
        else:
            custom_tail = raw_suffix if raw_suffix.startswith(",") else f", {raw_suffix}"
            suffix = (
                f"{custom_tail}, logarithmic sensor capture, uniform even exposure across subject and surfaces"
            )
        return f"{prefix}{cleaned}{suffix}"

    # Step 6: Legacy mode suffix (low_key vs warm_ambient vs studio_explicit vs universal)
    raw_suffix = (albedo_suffix or "").strip()
    if raw_suffix in LEGACY_DEFAULT_SUFFIXES or raw_suffix.lower() == "auto":
        exposure_mode = detect_photometric_exposure(master_prompt)
        resolved_suffix = PHOTOMETRIC_SUFFIXES[exposure_mode]
    else:
        resolved_suffix = raw_suffix if raw_suffix.startswith(",") else f", {raw_suffix}"

    return f"{cleaned}{resolved_suffix}"


class StudioPromptSanitizer:
    """
    Decouples zero-photon darkness, harsh cast shadows, and directional flash traps
    between Pass 1 (Albedo Draft) and Pass 2 (Lit Synthesis) using the Universal
    Semantic Scene Parser + Lighting Policy Resolver (RIG_OVERRIDE vs HYBRID_PRACTICAL).
    """

    DEFAULT_ALBEDO_SUFFIX = "auto"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "High fashion editorial of a young elegant woman holding a glossy black patent leather handbag, "
                        "wearing a structured Savile Row black wool coat, sleek swept-back dark hair, "
                        "neoclassical Parisian stone architecture background at night, shot on 35mm medium format, "
                        "50mm portrait lens, detailed fabric weave, authentic editorial campaign."
                    ),
                }),
            },
            "optional": {
                "lighting_policy": (["RIG_OVERRIDE", "HYBRID_PRACTICAL", "VERBATIM_RAW", "auto", True, False], {
                    "default": "RIG_OVERRIDE",
                }),
                "capture_guidance": (["Neutral Diffuse Reference", "Balanced Studio Capture", "Preserve Environmental Tone", "Legacy Log-C", True, False, "true", "false", "auto"], {
                    "default": "Neutral Diffuse Reference",
                }),
                "minimal_refinement_pass2": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON (Preserve Details)",
                    "label_off": "OFF (Pure Scene)",
                }),
                "albedo_suffix": ("STRING", {
                    "multiline": True,
                    "default": "auto",
                }),
                "pass1_log_acquisition": ("BOOLEAN", {
                    "default": True,
                    "label_on": "ON (Log-C Flat Capture)",
                    "label_off": "Legacy Diffuse",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pass1_albedo_prompt", "pass2_lit_prompt")
    FUNCTION = "sanitize"
    CATEGORY = "StudioLight/Utility"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def sanitize(
        self,
        master_prompt: str,
        albedo_suffix: str = DEFAULT_ALBEDO_SUFFIX,
        pass1_log_acquisition: bool = True,
        lighting_policy: str = "RIG_OVERRIDE",
        capture_guidance: str = "Neutral Diffuse Reference",
        minimal_refinement_pass2: bool = True,
        **kwargs,
    ) -> Tuple[str, str]:
        if str(lighting_policy).lower() in ("auto", "true", "false", "none", "") or lighting_policy not in ("RIG_OVERRIDE", "HYBRID_PRACTICAL", "VERBATIM_RAW"):
            lighting_policy = "RIG_OVERRIDE"
        if isinstance(capture_guidance, bool) or str(capture_guidance).lower() in ("true", "auto", "1"):
            capture_guidance = "Neutral Diffuse Reference"
        elif str(capture_guidance).lower() in ("false", "0"):
            capture_guidance = "Legacy Log-C"

        if lighting_policy == "VERBATIM_RAW":
            pass1_prompt = sanitize_pass1_prompt_universal(
                master_prompt,
                albedo_suffix=albedo_suffix,
                pass1_log_acquisition=pass1_log_acquisition,
            )
            pass2_prompt = master_prompt.strip()
            return (pass1_prompt, pass2_prompt)

        # Semantic Parser v2 Pipeline
        scene = parse_scene_description(master_prompt, lighting_policy=lighting_policy)

        # Determine exposure mode for photometric suffixes
        exposure_mode = detect_photometric_exposure(master_prompt)
        mode_suffix = PHOTOMETRIC_SUFFIXES.get(exposure_mode, "")

        # Pass 1 Albedo Prompt
        pass1_prompt = build_pass1_prompt_v2(
            scene,
            capture_guidance=capture_guidance,
            albedo_suffix=f"{mode_suffix}, {albedo_suffix}" if albedo_suffix != "auto" else mode_suffix,
        )

        # Pass 2 Lit Prompt
        pass2_prompt = build_pass2_prompt_v2(
            scene,
            lighting_policy=lighting_policy,
            minimal_refinement=minimal_refinement_pass2,
        )

        return (pass1_prompt, pass2_prompt)


NODE_CLASS_MAPPINGS = {
    "StudioPromptSanitizer": StudioPromptSanitizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioPromptSanitizer": "StudioLight Prompt Sanitizer (Auto-Albedo)",
}
