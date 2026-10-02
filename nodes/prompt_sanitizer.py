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

        # 2. Strip standalone high-specular / wet / glare tokens
        specular_tokens = [
            r'\bultra[- ]?glossy\b', r'\bhigh[- ]?gloss\b', r'\bglossy\b',
            r'\bpatent\b', r'\bshiny\b', r'\blacquered\b', r'\bvarnished\b',
            r'\bwet\b', r'\breflective\b', r'\bspecular\b', r'\bglare\b'
        ]
        for token in specular_tokens:
            cleaned = re.sub(token, '', cleaned, flags=re.IGNORECASE)

        # 3. Clean up punctuation artifacts & multi-spaces
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
