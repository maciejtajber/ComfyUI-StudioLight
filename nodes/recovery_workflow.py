"""Recovery adapters for the original StudioLight console and shader.

No learned material/sky/identity inference. Geometry is a bounded scene prior.
Additive plate uses source RGB as an appearance proxy, never calls it albedo.
Existing node interfaces and saved workflows remain unchanged.
"""
import copy
import inspect
import math
import torch
import torch.nn.functional as F
from .studio_light import StudioLightRig
from .preset_manager import get_all_presets_flattened
from .prompt_synthesizer import synthesize_high_signal_prompt
from .shader_engine import (render_studio_lighting, compute_light_shading,
                            azimuth_elevation_to_vector, resolve_aim_target, compute_cavity_ao_map)
from .color_utils import kelvin_to_rgb
from .nodes_transport import (StudioTransportGeometry, _image, _mask,
                             srgb_to_linear, linear_to_srgb, _heatmap)


class StudioLightConsole(StudioLightRig):
    @classmethod
    def INPUT_TYPES(cls):
        result = copy.deepcopy(super().INPUT_TYPES())
        for k in ('normals_optional', 'depth_optional'):
            result.get('optional', {}).pop(k, None)
        result['optional']['depth_relief_scale'][1]['tooltip'] = 'Legacy relief control; bounded metric geometry uses Space & Ambient instead.'
        return result
    RETURN_TYPES = ('SL_RIG', 'IMAGE', 'STRING')
    RETURN_NAMES = ('shared_rig', 'probe_preview', 'lighting_prompt')
    FUNCTION = 'prepare'
    CATEGORY = 'StudioLight/Recovery'

    def prepare(self, **kwargs):
        args = {k: p.default for k, p in inspect.signature(StudioLightRig.execute_rig).parameters.items()
                if p.default is not inspect.Parameter.empty}
        args.update(kwargs)
        args.pop('normals_optional', None)
        args.pop('depth_optional', None)
        preset = get_all_presets_flattened().get(args.get('preset'), {})
        args.update(preset)
        args['preset'] = 'Custom'  # Resolve ONCE; prompt, probe and transport share this result.
        lights = [{k: args[f'{name}_{k}'] for k in
                   ('enabled', 'azimuth', 'elevation', 'distance', 'intensity', 'kelvin', 'modifier')}
                  for name in ('key', 'fill', 'rim')]
        # Small procedural normal plate avoids creating a 1024px dummy scene.
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij')
        zz = (1-xx.square()-yy.square()).clamp_min(0).sqrt()
        sphere = (torch.stack((xx, -yy, zz), -1)[None]+1)/2
        _, probe = render_studio_lighting(sphere, depth=None, key_params=lights[0], fill_params=lights[1],
                    rim_params=lights[2], ambient_intensity=args['ambient_intensity'],
                    preview_resolution=256, invert_normals_x=False)
        prompt = synthesize_high_signal_prompt(*lights, target_engine=args.get('target_engine', 'Flux.1'))
        return {'schema': 1, 'lights': lights, 'settings': args}, probe, prompt


class StudioLightSpace:
    @classmethod
    def INPUT_TYPES(cls):
        result = copy.deepcopy(StudioTransportGeometry.INPUT_TYPES())
        for k in ('depth', 'subject_mask'):
            result.get('required', {}).pop(k, None)
            result.get('optional', {}).pop(k, None)
        return result
    RETURN_TYPES = ('SL_SPACE', 'FLOAT')
    RETURN_NAMES = ('shared_space', 'camera_fov')
    FUNCTION = 'prepare'
    CATEGORY = 'StudioLight/Recovery'
    def prepare(self, **kwargs):
        return dict(kwargs), float(kwargs.get("fov_deg", 50.))


class StudioLightGenerationPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'subject': ('STRING', {'forceInput': True}),
                             'lighting': ('STRING', {'forceInput': True}), 'space': ('SL_SPACE',)}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('same_prompt_for_both_passes',)
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Recovery'
    def run(self, subject, lighting, space):
        env = space.get('environment', 'Inside')
        ambient = space.get('ambient_lighting', 'In-Between')
        text = {'Dark': 'dark ambient surroundings with existing practical lights retained',
                'Bright': 'bright ambient surroundings', 'In-Between': 'balanced ambient surroundings'}[ambient]
        if env == 'Outside':
            receiver = 'Nearby solid surfaces receive fixture light; distant sky retains its ambient appearance.'
        else:
            size = float(space.get('space_size', 6))
            unit = space.get('space_measurement', 'Meters (Depth)')
            receiver = f'Enclosed setting, approximate space size {size:g} {unit}; nearby walls receive fixture light with distance falloff.'
        return (f'{subject.strip()}\nLighting direction: {lighting.strip()}\nEnvironment: {text}. {receiver}',)


class StudioLightSceneGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'depth': ('IMAGE',), 'space': ('SL_SPACE',), 'rig': ('SL_RIG',)},
                'optional': {'subject_mask': ('MASK',), 'receiver_mask': ('MASK',)}}
    RETURN_TYPES = ('SL_GEOMETRY', 'IMAGE', 'MASK', 'MASK', 'STRING')
    RETURN_NAMES = ('geometry', 'assumed_depth', 'subject_estimate', 'receiver_estimate', 'report')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Recovery'
    def run(self, depth, space, rig, subject_mask=None, receiver_mask=None):
        d = _image(depth, 'depth', (1, 3))[..., :1]
        if rig['settings'].get('invert_depth', False):
            d = 1-d
        if subject_mask is None:
            # Soft near-depth prior; NOT semantic subject segmentation.
            threshold = d.flatten(1).median(1).values[:, None, None, None]
            subject = ((d-threshold)/.15+.5).clamp(0, 1)
        else:
            subject = _mask(subject_mask, d)
        if receiver_mask is None:
            receiver = torch.ones_like(d)
            if space.get('environment') == 'Outside':
                # Conservative far-depth suppression, not a sky detector.
                receiver = (d/.08).clamp(0, 1)
        else:
            receiver = _mask(receiver_mask, d)
        geometry, preview, report = StudioTransportGeometry().run(
            d.expand(-1, -1, -1, 3), subject[..., 0], **space)
        geometry['receiver_mask'] = receiver
        return geometry, preview, subject[..., 0], receiver[..., 0], report


class StudioLightSharedTransport:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'normals': ('IMAGE',), 'geometry': ('SL_GEOMETRY',), 'rig': ('SL_RIG',)}}
    RETURN_TYPES = ('SL_FLASH', 'IMAGE', 'IMAGE', 'IMAGE')
    RETURN_NAMES = ('subject_and_wall_transport', 'diffuse_heatmap', 'specular_preview', 'visibility')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Recovery'
    def run(self, normals, geometry, rig):
        n = _image(normals, 'normals', (3,))*2-1
        settings = rig['settings']
        if settings.get('invert_normals_x', True): n[..., 0] *= -1
        if settings.get('invert_normals_y', False): n[..., 1] *= -1
        n[..., :2] *= max(float(settings.get('normal_boost', 1)), .5)
        n = F.normalize(n, dim=-1, eps=1e-7)
        z = geometry['depth_m'].to(n)
        if z.shape[:3] != n.shape[:3]:
            raise ValueError('Depth and normals must have the same batch and resolution')
        fx, fy, cx, cy = geometry['intrinsics']
        b, h, w, _ = n.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=n.device), torch.arange(w, device=n.device), indexing='ij')
        # Existing shader: +X right, +Y up, negative Z into scene.
        rays = torch.stack(((xx-cx)/fx, -(yy-cy)/fy, -torch.ones_like(xx)), -1).to(n)
        points = z*rays[None]
        depth = geometry['warped_depth'].to(n)
        vfov = math.degrees(2*math.atan(h/(2*fy)))
        diffuse, specular, visibility = [], [], []
        for i in range(b):
            p, d, ni = points[i:i+1], depth[i:i+1], n[i:i+1]
            split = float(d.median())
            aim = resolve_aim_target(p, d, ni, aim_mode=settings.get('aim_target', 'Upper'),
                                     d_split=split, device=n.device, dtype=n.dtype)
            diff, spec = torch.zeros_like(ni), torch.zeros_like(ni)
            unocc, blocked = torch.zeros_like(d), torch.zeros_like(d)
            for light in rig['lights']:
                if not light['enabled'] or light['intensity'] <= 0: continue
                result = compute_light_shading(ni, azimuth_elevation_to_vector(light['azimuth'], light['elevation']),
                    kelvin_to_rgb(light['kelvin']), light['intensity'], light['modifier'], d,
                    distance=light['distance'], return_separate=True, target_pos=aim,
                    vertical_fov_deg=vfov, precomputed_surface=(p, d, split), return_shadow_telemetry=True)
                diff += result[0]; spec += result[1]; unocc += result[2]; blocked += result[3]
            if settings.get('ao_anchoring', 0) > 0:
                ao = compute_cavity_ao_map(ni, strength=float(settings['ao_anchoring'])*2.5)
                diff *= ao; spec *= ao
            mask = geometry['receiver_mask'][i:i+1].to(n)
            diffuse.append(diff*mask); specular.append(spec*mask)
            visibility.append(torch.where(unocc>1e-8, 1-blocked/unocc.clamp_min(1e-8), torch.ones_like(unocc)))
        e, s, v = torch.cat(diffuse), torch.cat(specular), torch.cat(visibility)
        payload = {'schema': 1, 'diffuse': e, 'specular': s, 'units': 'gain', 'receiver_mask_applied': True}
        return payload, _heatmap(e), linear_to_srgb(s.clamp(0, 1)), v.expand(-1,-1,-1,3)


class StudioLightExtraLight:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'draft': ('IMAGE',), 'transport': ('SL_FLASH',),
               'extra_light_strength': ('FLOAT', {'default': 0., 'min': 0., 'max': 4., 'step': .05}),
               'specular_strength': ('FLOAT', {'default': 0., 'min': 0., 'max': 1., 'step': .05})}}
    RETURN_TYPES = ('IMAGE', 'IMAGE')
    RETURN_NAMES = ('proxy_for_pass_2', 'draft_vs_proxy')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Recovery'
    def run(self, draft, transport, extra_light_strength=0., specular_strength=0.):
        c = srgb_to_linear(draft)
        e, s = transport['diffuse'].to(c), transport['specular'].to(c)
        if e.shape != c.shape or s.shape != c.shape:
            raise ValueError('Draft and transport dimensions must match')
        if not math.isfinite(extra_light_strength) or extra_light_strength < 0 or not math.isfinite(specular_strength) or specular_strength < 0:
            raise ValueError('Light strengths must be finite and nonnegative')
        if extra_light_strength == 0:
            out = draft.clone()  # Exact identity, including display conversion.
        else:
            # Approximate appearance-weighted addition. No manufactured skin floor,
            # no ambient subtraction, no claimed mirror-reflection replacement.
            out = linear_to_srgb((c + extra_light_strength*(c*e + specular_strength*s)).clamp(0, 1))
        return out, torch.cat((draft, out), dim=2)


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (StudioLightConsole, StudioLightSpace,
    StudioLightGenerationPrompt, StudioLightSceneGeometry, StudioLightSharedTransport, StudioLightExtraLight)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'StudioLightConsole': 'StudioLight · Original Console — Before Pass 1',
    'StudioLightSpace': 'StudioLight · Space & Ambient',
    'StudioLightGenerationPrompt': 'StudioLight · Light-First Prompt',
    'StudioLightSceneGeometry': 'StudioLight · Scene Geometry',
    'StudioLightSharedTransport': 'StudioLight · Shared Rig — Subject & Walls',
    'StudioLightExtraLight': 'StudioLight · Extra Light (0 = Preserve Draft)',
}
