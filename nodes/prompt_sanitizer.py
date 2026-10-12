import re

class StudioPromptSanitizer:
    """
    Decouples high-specularity tokens between Pass 1 (Albedo Draft)
    and Pass 2 (Lit Synthesis) to eliminate pre-baked specular blobs.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "master_prompt": ("STRING", {
                    "multiline": True,
                    "default": "editorial portrait of a woman holding a glossy black patent leather handbag, tailored wool coat"
                }),
            },
            "optional": {
                "albedo_suffix": ("STRING", {
                    "multiline": True,
                    "default": ", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pass1_albedo_prompt", "pass2_lit_prompt")
    FUNCTION = "sanitize"
    CATEGORY = "StudioLight/Utility"

    DEFAULT_ALBEDO_SUFFIX = ", flat overcast diffuse daylight, soft even shadowless lighting, neutral beauty catalog"

    def sanitize(self, master_prompt: str, albedo_suffix: str = DEFAULT_ALBEDO_SUFFIX):
        cleaned = master_prompt

        # 1. Transform compound phrases
        cleaned = re.sub(r'\b(ultra[- ]?glossy|high[- ]?gloss|glossy|patent|lacquered)\s+leather\b', 'smooth leather', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(liquid\s+chrome|chrome)\b', 'matte brushed metal', cleaned, flags=re.IGNORECASE)

        # 2. Strip directional / rim-light / sunlight / studio-fixture phrases & clauses
        # Note: No arbitrary \w+ wildcards are permitted between adjectives and lighting nouns so
        # adjacent subject phrases (e.g. "blue dress", "amber jewelry", "warm glow") are never swallowed.
        light_adj = r'(?:dramatic|harsh|hard|direct|strong|intense|warm|cool|blue|cyan|amber|golden|neon|cinematic|soft|subtle|bright|deep|cast|directional)'
        light_noun_all = (
            r'(?:rim[- ]?light(?:ing)?|back[- ]?light(?:ing)?|back[- ]?lit|sunlight|'
            r'key[- ]?light(?:ing)?|side[- ]?light(?:ing)?|spot[- ]?light(?:ing)?|'
            r'hair[- ]?light(?:ing)?|kicker(?:\s+light)?|lighting|shadows)'
        )
        light_noun_bare = (
            r'(?:rim[- ]?light(?:ing)?|back[- ]?light(?:ing)?|back[- ]?lit|sunlight|'
            r'key[- ]?light(?:ing)?|side[- ]?light(?:ing)?|spot[- ]?light(?:ing)?|'
            r'hair[- ]?light(?:ing)?|kicker\s+light|'
            r'chiaroscuro(?:\s+lighting|\s+shadows)?|rembrandt\s+lighting|'
            r'split\s+lighting|volumetric\s+(?:god\s+)?rays)'
        )
        dir_tail = r'(?:\s+from\s+(?:the\s+)?(?:left|right|behind|above|below|side|rear|front))?'
        lead_prep = r'(?:\b(?:in|under|with|lit\s+by)\s+(?:the\s+)?)?'

        directional_patterns = [
            rf'{lead_prep}\b(?:{light_adj}\s+){{1,3}}{light_noun_all}\b{dir_tail}',
            rf'{lead_prep}\b{light_noun_bare}\b{dir_tail}',
        ]
        for pat in directional_patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

        # 3. Strip standalone high-specular / wet / glare tokens
        specular_tokens = [
            r'\bultra[- ]?glossy\b', r'\bhigh[- ]?gloss\b', r'\bglossy\b',
            r'\bpatent\b', r'\bshiny\b', r'\blacquered\b', r'\bvarnished\b',
            r'\bwet\b', r'\breflective\b', r'\bspecular\b', r'\bglare\b'
        ]
        for token in specular_tokens:
            cleaned = re.sub(token, '', cleaned, flags=re.IGNORECASE)

        # 4. Clean up punctuation artifacts & multi-spaces
        cleaned = re.sub(r'\s*,\s*,+', ',', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip(' ,')

        # 4. Construct Pass 1 Albedo Prompt
        suffix = albedo_suffix.strip()
        if suffix and not suffix.startswith(','):
            suffix = ', ' + suffix
        pass1_prompt = f"{cleaned}{suffix}"

        # 5. Pass 2 retains 100% of master user prompt
        pass2_prompt = master_prompt.strip()

        return (pass1_prompt, pass2_prompt)


NODE_CLASS_MAPPINGS = {
    "StudioPromptSanitizer": StudioPromptSanitizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StudioPromptSanitizer": "StudioLight Prompt Sanitizer (Auto-Albedo)",
}
