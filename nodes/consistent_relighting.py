"""Explicit source + studio-light compositing. Copyright 2026 Maciej Tajber. MIT.

See docs/CONSISTENT_RELIGHTING_CONTRACT.md. No material/emitter inference from RGB.
"""
import json
import re
from collections import deque
import torch
import torch.nn.functional as F
from .semantic_parser import parse_scene_description, build_pass1_prompt_v2, build_pass2_prompt_v2
from .photometric_composite import srgb_to_linear, filmic_reinhard_tonemap
from .photometric_support import scalar


def encode_display(hdr, baseline=None):
    x = filmic_reinhard_tonemap(hdr.clamp_min(0))
    if baseline is not None:
        # Preserve already display-referred source highlights at zero added
        # light. Tone-map only the added response relative to that baseline.
        x = baseline.clamp(0, 1) + x - filmic_reinhard_tonemap(baseline.clamp_min(0))
    x = x.clamp(0, 1)
    return torch.where(x <= .0031308, x * 12.92, 1.055 * x.clamp_min(1e-12).pow(1 / 2.4) - .055).clamp(0, 1)


def scene_profile(prompt, environment='Auto', ambient_intent='Auto'):
    p = prompt.lower()
    practicals = sorted(set(re.findall(r'\b(?:torches|torch|flames?|candles?|lanterns?|neon|lasers?|campfire|bonfire|smartphone|phone screen|glowing screen|glowing sign)\b', p)))
    indoor = bool(re.search(r'\b(?:inside|indoor|interior|cave|cavern|room|club|nightclub|rave|tunnel|warehouse)\b', p))
    outdoor = bool(re.search(r'\b(?:outside|outdoor|open sky|sky|lake|ocean|beach|forest|street|alley|terrace|courtyard)\b', p))
    inferred = 'Mixed' if indoor and outdoor else 'Enclosed' if indoor else 'Open' if outdoor else 'Unknown'
    choices = ('Auto', 'Open', 'Enclosed', 'Mixed')
    if environment not in choices:
        raise ValueError('Unknown environment')
    if ambient_intent not in ('Auto', 'Bright', 'Dark'):
        raise ValueError('Unknown ambient intent')
    # Do not classify a bright room as night merely because it contains black clothes/hair.
    dark = bool(re.search(r'\b(?:night|nighttime|midnight|dusk|twilight|moonless|low[- ]key|dimly lit|pitch[- ]black|darkness|dark(?: underground| techno| stone| enclosed){0,2} (?:cave|cavern|room|club|alley))\b', p))
    bright = bool(re.search(r'\b(?:daylight|daytime|brightly lit|high[- ]key|sunny|noon)\b', p))
    intent = ambient_intent if ambient_intent != 'Auto' else 'Dark' if dark else 'Bright' if bright else 'Unspecified'
    policy = 'HYBRID_PRACTICAL' if practicals else 'RIG_OVERRIDE'
    scene = parse_scene_description(prompt, lighting_policy=policy)
    capture = 'Preserve Environmental Tone' if intent == 'Dark' else 'Neutral Diffuse Reference'
    p1 = build_pass1_prompt_v2(scene, capture_guidance=capture)
    p2 = build_pass2_prompt_v2(scene, lighting_policy=policy)
    contract = {'schema_version': 1, 'original_prompt': prompt, 'environment': inferred if environment == 'Auto' else environment,
                'environment_inferred': inferred, 'ambient_intent': intent, 'practicals': practicals,
                'lighting_policy': policy, 'material_categories': scene.material_categories,
                'geometry_scale': 'assumed', 'inference_is_estimate': True}
    return contract, p1, p2


def _rgb(x, name, device, display=False):
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[-1] != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError(f'{name} must be nonempty BHWC floating RGB')
    x = x.to(device=device, dtype=torch.float32)
    if not torch.isfinite(x).all() or (x < 0).any() or (display and (x > 1).any()):
        raise ValueError(f'{name} must be finite and nonnegative' + (' sRGB in [0,1]' if display else ''))
    return x


def _mask(x, name, ref):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f'{name} must be a MASK tensor')
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    if x.ndim != 4 or x.shape[-1] != 1 or min(x.shape) < 1:
        raise ValueError(f'{name} must be BHW or BHW1')
    x = x.to(device=ref.device, dtype=torch.float32)
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError(f'{name} must be finite in [0,1]')
    return x


def far_background_mask(depth, subject):
    """Conservative top-connected far-depth estimate. Not semantic sky detection."""
    B, H, W, _ = depth.shape
    size = (min(H, 256), min(W, 256))
    d = F.interpolate(depth.permute(0, 3, 1, 2), size=size, mode='nearest')[:, 0].cpu()
    s = F.interpolate(subject.permute(0, 3, 1, 2), size=size, mode='nearest')[:, 0].cpu()
    out = torch.zeros_like(d)
    for b in range(B):
        # Constant depth does not establish infinity/sky.
        if float(d[b].max() - d[b].min()) < .02:
            continue
        candidate = ((d[b] <= .08) & (s[b] < .2)).tolist()
        visited = [[False] * size[1] for _ in range(size[0])]
        q = deque()
        for x in range(size[1]):
            if candidate[0][x]:
                visited[0][x] = True
                q.append((0, x))
        while q:
            y, x = q.popleft()
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < size[0] and 0 <= xx < size[1] and candidate[yy][xx] and not visited[yy][xx]:
                    visited[yy][xx] = True
                    q.append((yy, xx))
        out[b] = torch.tensor(visited, dtype=out.dtype)
    return F.interpolate(out[:, None].to(depth.device), size=(H, W), mode='nearest').permute(0, 2, 3, 1)


def render_consistent(source, diffuse_base, telemetry, scene_contract,
                      operation='Add studio light', studio_strength=1., specular_strength=0.,
                      scene_exposure=0., source_preservation=1., protect_far_background=True,
                      subject_mask=None, sky_mask=None, receiver_mask=None):
    """Return display proxy, diffuse/specular previews, protected mask, HDR passes, report."""
    if not isinstance(scene_contract, dict) or scene_contract.get('schema_version') != 1:
        raise ValueError('Expected explicit StudioLight scene contract schema 1')
    if not isinstance(telemetry, dict) or telemetry.get('schema_version') != 2:
        raise ValueError('Expected explicit rig telemetry schema 2; connect rig telemetry output')
    if telemetry.get('depth_convention') != 'relative_near_one':
        raise ValueError('Expected relative_near_one depth')
    if operation not in ('Add studio light', 'Replace subject light'):
        raise ValueError('Unknown relighting operation')
    strength = scalar(studio_strength, 'studio_strength', 0, 4)
    spec_strength = scalar(specular_strength, 'specular_strength', 0, 3)
    exposure = scalar(scene_exposure, 'scene_exposure', -8, 4)
    preservation = scalar(source_preservation, 'source_preservation', 0, 1)
    if not isinstance(source, torch.Tensor):
        raise ValueError('source must be an IMAGE tensor')
    device = source.device
    images = {'source': _rgb(source, 'source', device, True), 'base': _rgb(diffuse_base, 'diffuse_base', device, True)}
    for key in ('surface_direct_diffuse', 'surface_direct_specular', 'surface_ambient_rgb'):
        if key not in telemetry:
            raise ValueError(f'Missing full-surface telemetry {key}; update the rig package together with this node')
        images[key] = _rgb(telemetry[key], key, device)
    masks = {}
    for key, x in (('subject', subject_mask), ('sky', sky_mask), ('receiver', receiver_mask)):
        if x is not None:
            masks[key] = _mask(x, key, images['source'])
    if 'd_norm' in telemetry:
        masks['depth'] = _mask(telemetry['d_norm'], 'depth', images['source'])
    size = images['source'].shape[1:3]
    values = [*images.values(), *masks.values()]
    if any(x.shape[1:3] != size for x in values):
        raise ValueError('All surface maps must align exactly with the source; no hidden resizing')
    batch = max(x.shape[0] for x in values)
    if any(x.shape[0] not in (1, batch) for x in values):
        raise ValueError('Batches must match or be singleton')
    images = {k: x.expand(batch, *x.shape[1:]) for k, x in images.items()}
    masks = {k: x.expand(batch, *x.shape[1:]) for k, x in masks.items()}
    zero = torch.zeros_like(images['source'][..., :1])
    warnings = []
    if operation == 'Replace subject light' and 'subject' not in masks:
        masks['subject'] = torch.ones_like(images['source'][..., :1])
        warnings.append('No subject_mask provided for Replace subject light; defaulting to full-frame subject.')
    subject = masks.get('subject', zero)
    protected = masks.get('sky', zero)
    env = scene_contract.get('environment', 'Unknown')
    if sky_mask is None and protect_far_background and env in ('Open', 'Mixed'):
        if 'depth' in masks:
            protected = far_background_mask(masks['depth'], subject)
            warnings.append('Far-background mask is a top-connected depth estimate; inspect preview, not semantic sky truth.')
        else:
            warnings.append('No depth: automatic far-background protection unavailable.')
    elif env == 'Unknown':
        warnings.append('Environment unknown; no automatic sky label. Set environment or supply a sky mask if needed.')
    receiver = (1-protected) * masks.get('receiver', torch.ones_like(zero))
    if operation == 'Replace subject light':
        receiver = receiver * subject
    source_linear = srgb_to_linear(images['source']) * (2. ** exposure)
    base = srgb_to_linear(images['base'])
    diffuse = base * images['surface_direct_diffuse'] * strength * receiver
    specular = images['surface_direct_specular'] * (spec_strength / .25) * strength * receiver
    ambient = base * images['surface_ambient_rgb'] * receiver
    scene = source_linear if operation == 'Add studio light' else source_linear * (1-subject + preservation*subject)
    hdr = scene + diffuse + specular + ambient
    if not torch.isfinite(hdr).all():
        raise ValueError('Relighting overflowed; lower HDR light values or exposure')
    passes = {'schema_version': 1, 'color_space': 'linear_RGB', 'units': 'relative_display_linear',
              'source_scene': scene, 'new_diffuse': diffuse, 'new_specular': specular,
              'new_ambient': ambient, 'combined': hdr, 'receiver_mask': receiver, 'protected_background': protected,
              'operation': operation, 'geometry_scale': 'assumed'}
    if operation == 'Replace subject light' and preservation > 0:
        warnings.append('Source preservation retains old subject lighting/reflections as well as practicals; it is not emitter detection.')
    if operation == 'Replace subject light' and preservation == 0 and scene_contract.get('practicals'):
        warnings.append('Subject practicals/spill may be removed at source_preservation=0.')
    report = json.dumps({'contract': 1, 'operation': operation, 'environment': env,
                         'practicals_preserved_by': 'original scene appearance' if operation == 'Add studio light' else 'source_preservation',
                         'warnings': warnings, 'visual_gpu_acceptance': 'pending'}, indent=2)
    return encode_display(hdr, scene), encode_display(diffuse), encode_display(specular), protected[..., 0], passes, report


class StudioLightSceneContract:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'master_prompt': ('STRING', {'multiline': True}),
                             'environment': (['Auto', 'Open', 'Enclosed', 'Mixed'],),
                             'ambient_intent': (['Auto', 'Bright', 'Dark'],)}}
    RETURN_TYPES = ('STUDIOLIGHT_SCENE_CONTRACT', 'STRING', 'STRING', 'STRING')
    RETURN_NAMES = ('scene_contract', 'pass1_capture_prompt', 'pass2_scene_prompt', 'report')
    FUNCTION = 'build'
    CATEGORY = 'StudioLight/Consistent'

    def build(self, master_prompt, environment, ambient_intent):
        contract, p1, p2 = scene_profile(master_prompt, environment, ambient_intent)
        return contract, p1, p2, json.dumps(contract, indent=2)


class StudioLightConsistentComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'source': ('IMAGE',), 'diffuse_base': ('IMAGE',),
                             'telemetry': ('STUDIOLIGHT_TELEMETRY',), 'scene_contract': ('STUDIOLIGHT_SCENE_CONTRACT',),
                             'operation': (['Add studio light', 'Replace subject light'],),
                             'studio_strength': ('FLOAT', {'default': 1., 'min': 0., 'max': 4., 'step': .05}),
                             'specular_strength': ('FLOAT', {'default': 0., 'min': 0., 'max': 3., 'step': .05}),
                             'scene_exposure': ('FLOAT', {'default': 0., 'min': -8., 'max': 4., 'step': .1}),
                             'source_preservation': ('FLOAT', {'default': 1., 'min': 0., 'max': 1., 'step': .05, 'tooltip': 'Replace subject light only: retains original subject appearance, including practical illumination and old shadows. Add studio light always retains the source scene.'}),
                             'protect_far_background': ('BOOLEAN', {'default': True})},
                'optional': {'subject_mask': ('MASK',), 'sky_mask': ('MASK',), 'receiver_mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'IMAGE', 'MASK', 'STUDIOLIGHT_PASSES', 'STRING')
    RETURN_NAMES = ('lit_proxy', 'new_diffuse_preview', 'new_specular_preview', 'protected_background', 'linear_HDR_passes', 'report')
    FUNCTION = 'compose'
    CATEGORY = 'StudioLight/Consistent'

    @torch.inference_mode()
    def compose(self, **kwargs):
        return render_consistent(**kwargs)


NODE_CLASS_MAPPINGS = {'StudioLightSceneContract': StudioLightSceneContract,
                       'StudioLightConsistentComposite': StudioLightConsistentComposite}
NODE_DISPLAY_NAME_MAPPINGS = {'StudioLightSceneContract': 'StudioLight Scene Contract',
                              'StudioLightConsistentComposite': 'StudioLight Consistent Relighting'}
