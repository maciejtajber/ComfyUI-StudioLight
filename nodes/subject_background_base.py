"""Automatic subject from De-Light albedo; background from diagnostic A*S+R."""
import torch
import torch.nn.functional as F

from .photometric_support import image_tensor, scalar


def _linear(value):
    return torch.where(value <= 0.04045, value / 12.92,
                       ((value + 0.055) / 1.055).pow(2.4))


def _srgb(value):
    value = value.clamp_min(0)
    return torch.where(value <= 0.0031308, value * 12.92,
                       1.055 * value.pow(1 / 2.4) - 0.055)


def _validated_images(subject_base, background):
    if not isinstance(subject_base, torch.Tensor):
        raise ValueError('subject_base must be an IMAGE tensor')
    subject = image_tensor(subject_base, 'subject_base', subject_base.device)
    background = image_tensor(background, 'background', subject.device)
    if background.shape != subject.shape:
        raise ValueError('subject_base and background must have identical batch and image dimensions')
    for name, value in [('subject_base', subject), ('background', background)]:
        if bool((value < 0).any() or (value > 1).any()):
            raise ValueError(f'{name} must be an sRGB IMAGE in [0,1]')
    return subject, background


def combine_subject_background(subject_base, background, subject_mask, edge_softness=0):
    """Copy each source exactly in its region; blend only the subject matte edge."""
    subject, background = _validated_images(subject_base, background)
    radius = scalar(edge_softness, 'edge_softness', 0, 16)
    if radius != int(radius):
        raise ValueError('edge_softness must be a whole number')
    radius = int(radius)
    if not isinstance(subject_mask, torch.Tensor):
        raise ValueError('subject_mask must be a MASK tensor')
    if subject_mask.ndim == 3:
        subject_mask = subject_mask.unsqueeze(-1)
    mask = image_tensor(subject_mask, 'subject_mask', subject.device, (1,))
    if mask.shape[0] not in (1, subject.shape[0]) or mask.shape[1:3] != subject.shape[1:3]:
        raise ValueError('subject_mask must match image dimensions and batch (or batch=1)')
    if bool((mask < 0).any() or (mask > 1).any()):
        raise ValueError('subject_mask must be in [0,1]')
    mask = mask.expand(*subject.shape[:3], 1)
    # Remove low-confidence background speckles and make confident subject pixels exact copies.
    mask = ((mask - 0.05) / 0.90).clamp(0, 1)
    mask = mask * mask * (3 - 2 * mask)
    if radius:
        mask = F.avg_pool2d(F.pad(mask.permute(0, 3, 1, 2), (radius,) * 4,
                                 mode='replicate'), 2 * radius + 1, stride=1).permute(0, 2, 3, 1)
    blended = _srgb(_linear(subject) * mask + _linear(background) * (1 - mask)).clamp(0, 1)
    blended = torch.where(mask == 1, subject, blended)
    blended = torch.where(mask == 0, background, blended)
    return blended, mask[..., 0]


class StudioLightSubjectBackgroundBase:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'subject_base': ('IMAGE', {'tooltip': 'Marigold albedo_srgb: retained on the subject.'}),
            'background': ('IMAGE', {'tooltip': 'Marigold uncalibrated_reconstruction_srgb (A*S+R).'}),
        }, 'optional': {
            'reference_image': ('IMAGE', {'tooltip': 'Optional aligned image for subject detection; defaults to subject_base.'}),
            'subject_mask': ('MASK', {'tooltip': 'Optional automatic mask from another detector; white=subject.'}),
            'edge_softness': ('INT', {'default': 0, 'min': 0, 'max': 16}),
        }}

    RETURN_TYPES = ('IMAGE', 'MASK', 'STRING')
    RETURN_NAMES = ('combined_base', 'subject_mask', 'report')
    FUNCTION = 'combine'
    CATEGORY = 'StudioLight/Experimental'
    DESCRIPTION = ('Automatically retain the subject from De-Light Base and the whole background '
                   'from A*S+R. Uses BiRefNet-general from comfyui-rmbg, including non-human '
                   'subjects. No water/color rules or painted masks. Detection is an estimate; '
                   'inspect the mask for transparent objects, reflections and multiple subjects.')

    def __init__(self):
        self._segmenter = None

    def _automatic_subject_mask(self, reference):
        import nodes as comfy_nodes
        detector = getattr(comfy_nodes, 'NODE_CLASS_MAPPINGS', {}).get('BiRefNetRMBG')
        if detector is None:
            raise RuntimeError('Automatic subject detection requires BiRefNetRMBG from comfyui-rmbg. '
                               'Install that pack or connect an automatic subject_mask. '
                               'No replacement or empty mask was silently substituted.')
        if self._segmenter is None:
            self._segmenter = detector()
        precision = torch.get_float32_matmul_precision()
        try:
            # The third-party loader initializes weights and changes global precision.
            # Keep those side effects from changing later sampling behavior.
            with torch.random.fork_rng(devices=[]):
                result = self._segmenter.process_image(
                    reference.detach().to('cpu'), model='BiRefNet-general',
                    mask_blur=0, mask_offset=0, invert_output=False,
                    refine_foreground=False, background='Alpha', background_color='#222222')
        finally:
            torch.set_float32_matmul_precision(precision)
            wrapper = getattr(self._segmenter, 'model', None)
            cleanup = getattr(wrapper, 'clear_model', None)
            if callable(cleanup):
                cleanup()
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RuntimeError('The subject detector did not return a mask')
        return result[1]

    @torch.inference_mode()
    def combine(self, subject_base, background, reference_image=None,
                subject_mask=None, edge_softness=0):
        subject, background = _validated_images(subject_base, background)
        method = 'connected mask'
        if subject_mask is None:
            reference = subject if reference_image is None else image_tensor(
                reference_image, 'reference_image', subject.device)
            if reference.shape != subject.shape or bool((reference < 0).any() or (reference > 1).any()):
                raise ValueError('reference_image must be an aligned sRGB IMAGE matching subject_base')
            subject_mask = self._automatic_subject_mask(reference)
            method = 'automatic BiRefNet-general'
        image, mask = combine_subject_background(subject, background, subject_mask, edge_softness)
        subject_percent = 100 * float((mask == 1).float().mean())
        background_percent = 100 * float((mask == 0).float().mean())
        report = (f'{method}: {subject_percent:.1f}% exact De-Light subject; '
                  f'{background_percent:.1f}% exact A*S+R background; remaining pixels are blended edges. '
                  'Source copies are exact in confident regions. The detected boundary is estimated. '
                  'The background retains its original lighting and reflections. '
                  'IMAGE inputs are sRGB; edge blending is performed in linear RGB.')
        return image, mask, report


NODE_CLASS_MAPPINGS = {'StudioLightSubjectBackgroundBase': StudioLightSubjectBackgroundBase}
NODE_DISPLAY_NAME_MAPPINGS = {'StudioLightSubjectBackgroundBase': 'StudioLight Automatic Subject / Background Base'}
