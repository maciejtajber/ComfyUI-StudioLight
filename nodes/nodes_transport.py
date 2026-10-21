"""Conservative linear-RGB relighting kernels. Layout: BHWC, float32.

These kernels do not infer identity, metric scale, sky masks, or true albedo.
No automatic resizing, color conversion, tone mapping, or strength normalization.
"""
import math
import torch
import torch.nn.functional as F
from .color_utils import kelvin_to_rgb


def _image(x, name, channels):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f'{name} must be a torch.Tensor')
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.ndim == 3:
        x = x.unsqueeze(-1)
    elif x.ndim == 4 and (1 in channels) and x.shape[-1] in (3, 4) and 3 not in channels:
        x = x[..., :3].mean(dim=-1, keepdim=True)
    elif x.ndim == 4 and (3 in channels) and x.shape[-1] == 1 and 1 not in channels:
        x = x.expand(-1, -1, -1, 3)
    elif x.ndim == 4 and x.shape[1] in (1, 3, 4) and x.shape[-1] not in channels and x.shape[1] in channels:
        x = x.permute(0, 2, 3, 1)

    if x.ndim != 4 or x.shape[-1] not in channels:
        raise ValueError(f'{name} must be BHWC with channels {channels}, got shape {tuple(x.shape)}')
    if min(x.shape) < 1 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f'{name} must be nonempty, finite floating point')
    return x.float()


def _field(x, ref, name, channels=(1, 3), low=0., high=None):
    x = _image(x, name, channels)
    if x.device != ref.device:
        x = x.to(device=ref.device)
    if x.shape[1:3] != ref.shape[1:3]:
        x_t = x.permute(0, 3, 1, 2)
        x_t = F.interpolate(x_t, size=ref.shape[1:3], mode='bilinear', align_corners=False)
        x = x_t.permute(0, 2, 3, 1)
    if x.shape[0] not in (1, ref.shape[0]):
        raise ValueError(f'{name}: only singleton batch broadcasts')
    if (x < low).any() or (high is not None and (x > high).any()):
        x = x.clamp(min=low, max=high if high is not None else float('inf'))
    return x.expand(ref.shape[0], -1, -1, -1)


def _number(x, name, low=0., strictly=False):
    x = float(x)
    if not math.isfinite(x) or (x <= low if strictly else x < low):
        raise ValueError(f'{name} must be finite and {">" if strictly else ">="} {low}')
    return x


def _luma(x):
    return (x * x.new_tensor([.2126, .7152, .0722])).sum(-1, keepdim=True)


def estimate_surface_reflectance(rgb, ambient_field, mask, *, prior,
                                 field_is_calibrated=False, confidence=None,
                                 regularization=.01, noise_floor=.002,
                                 max_gain=8.):
    """Regularized inversion of rgb = rho * ambient_field.

    rgb: linear RGB appearance, not sRGB. ambient_field: calibrated incident
    diffuse transfer E_ambient/pi in matching exposure units, NOT background RGB.
    prior: aligned material reflectance [0,1], mandatory; a flat skin floor is
    not an acceptable prior. Without a calibrated field, return prior and
    unresolved=True in mask. Missing information is not manufactured.

    Ridge estimate (a*c + lambda*p)/(a*a + lambda), blended toward p at low
    signal or ill-conditioned channels. Out-of-range inversion falls back to
    prior for the whole RGB pixel, avoiding per-channel hue clipping.
    confidence is caller evidence, not a learned probability.
    """
    rgb = _image(rgb, 'rgb', (3,))
    if (rgb < 0).any():
        raise ValueError('rgb must be nonnegative')
    m = _field(mask, rgb, 'mask', (1,), high=1)
    p = _field(prior, rgb, 'prior', (3,), high=1)
    lam = _number(regularization, 'regularization', strictly=True)
    floor = _number(noise_floor, 'noise_floor', strictly=True)
    cap = _number(max_gain, 'max_gain', low=1)
    if not field_is_calibrated:
        return {'reflectance': p.clone(), 'confidence': torch.zeros_like(m),
                'unresolved': m > 0}
    a = _field(ambient_field, rgb, 'ambient_field')
    q = torch.ones_like(m) if confidence is None else _field(confidence, rgb, 'confidence', (1,), high=1)
    raw = (a * rgb + lam * p) / (a.square() + lam)
    valid = ((raw >= 0) & (raw <= 1)).all(-1, keepdim=True)
    # Require every channel to have usable illumination: a green-only field
    # provides no evidence about red/blue reflectance.
    conditioning = (a.amin(-1, keepdim=True) * cap).clamp(0, 1)
    signal = (_luma(rgb) / floor - 1).clamp(0, 1)
    weight = q * conditioning * signal * valid
    rho = p + m * weight * (raw - p)
    return {'reflectance': rho, 'confidence': weight * m,
            'unresolved': (m > 0) & (weight < .5)}


def compute_cavity_wall_transport(depth, normals, light_pos, light_intensity, *,
                                 intrinsics, depth_anchors, receiver_mask,
                                 visibility, min_distance_m=.1,
                                 depth_model='inverse'):
    """Direct point-light transport, NOT cavity bounce/global illumination.

    Camera coordinates: +X right, +Y down, +Z forward; normals signed and facing
    out of the receiver. Camera-facing normal is (0,0,-1). Decode and transform
    DSINE conventions before calling. depth: BHW1, relative near=1.
    depth_anchors=((relative_far, metres_far),(relative_near, metres_near)).
    Two anchors are REQUIRED; one does not fix both affine scale and shift.
    inverse: 1/Z=a*depth+b. linear: Z=a*depth+b, explicit alternative assumption.
    Intrinsics=(fx,fy,cx,cy), calibrated for this exact image resolution.
    Light intensity: RGB radiant intensity, shape (3,) or (B,3); not total flux.
    Isotropic total flux Phi converts to I=Phi/(4*pi); strobes need beam data.
    receiver_mask excludes sky/air; visibility is supplied, not inferred here.
    Depth outside anchors is excluded rather than flattened to an endpoint.
    """
    d = _image(depth, 'depth', (1,))
    if (d < 0).any() or (d > 1).any():
        raise ValueError('depth must be in [0,1]')
    n = _field(normals, d, 'normals', (3,), low=-1, high=1)
    m = _field(receiver_mask, d, 'receiver_mask', (1,), high=1)
    v = _field(visibility, d, 'visibility', (1,), high=1)
    rmin = _number(min_distance_m, 'min_distance_m', strictly=True)
    (df, zf), (dn, zn) = depth_anchors
    df, zf, dn, zn = map(float, (df, zf, dn, zn))
    if not all(map(math.isfinite, (df, zf, dn, zn))) or not (0 <= df < dn <= 1 and 0 < zn < zf):
        raise ValueError('anchors require 0<=far_depth<near_depth<=1 and 0<near_m<far_m')
    fx, fy, cx, cy = map(float, intrinsics)
    if not all(map(math.isfinite, (fx, fy, cx, cy))) or min(fx, fy) <= 0:
        raise ValueError('intrinsics require positive focal lengths and finite principal point')
    t = ((d - df) / (dn - df)).clamp(0, 1)
    if depth_model == 'inverse':
        z = ((1-t)/zf + t/zn).reciprocal()
    elif depth_model == 'linear':
        z = (1-t)*zf + t*zn
    else:
        raise ValueError('depth_model must be inverse or linear')
    b, h, w, _ = d.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=d.device), torch.arange(w, device=d.device), indexing='ij')
    rays = torch.stack(((xx-cx)/fx, (yy-cy)/fy, torch.ones_like(xx)), -1).to(d.dtype)
    points = z * rays.unsqueeze(0)
    def vector(x, name, nonnegative=False):
        x = torch.as_tensor(x, dtype=d.dtype, device=d.device)
        if x.shape == (3,):
            x = x.unsqueeze(0)
        if x.shape not in ((1, 3), (b, 3)) or not torch.isfinite(x).all() or (nonnegative and (x < 0).any()):
            raise ValueError(f'{name} must be finite (3,) or (B,3) with valid values')
        return x.reshape(-1, 1, 1, 3)
    delta = vector(light_pos, 'light_pos') - points
    r2 = delta.square().sum(-1, keepdim=True)
    direction = delta / r2.sqrt().clamp_min(1e-8)
    length = torch.linalg.vector_norm(n, dim=-1, keepdim=True)
    unit_n = n / length.clamp_min(1e-8)
    valid = (d >= df) & (d <= dn) & (length > 1e-6) & (r2 > 1e-12)
    cosine = (unit_n * direction).sum(-1, keepdim=True).clamp_min(0)
    irradiance = vector(light_intensity, 'light_intensity', True) * cosine * m * v * valid / r2.clamp_min(rmin*rmin)
    return {'irradiance': irradiance, 'depth_m': z, 'distance_m': r2.sqrt(),
            'valid': valid, 'near_field_clamped': r2 < rmin*rmin}


def additive_flash_proxy(ambient, reflectance, direct_diffuse, direct_specular, mask,
                         *, diffuse_units, strength=1., exposure_scale=1.):
    """Add flash exactly once, preserving ambient when strength=0.

    direct_diffuse='irradiance': multiply rho*E/pi. exposure_scale converts
    resulting radiance and specular radiance to ambient's linear image units.
    direct_diffuse='gain': input already represents diffuse transfer; NO /pi.
    direct_specular must already include the material BRDF and use matching
    units. Inputs must EXCLUDE ambient and be computed at unit strength.
    Supply a combined receiver mask for foreground AND walls if lighting both.
    No clamping: HDR must survive until the explicit display/tone-map stage.
    """
    a = _image(ambient, 'ambient', (3,))
    if (a < 0).any():
        raise ValueError('ambient must be nonnegative')
    rho = _field(reflectance, a, 'reflectance', (3,), high=1)
    e = _field(direct_diffuse, a, 'direct_diffuse')
    s = _field(direct_specular, a, 'direct_specular', (3,))
    m = _field(mask, a, 'mask', (1,), high=1)
    k = _number(strength, 'strength')
    exposure = _number(exposure_scale, 'exposure_scale')
    if diffuse_units not in ('gain', 'irradiance'):
        raise ValueError('diffuse_units must explicitly be gain or irradiance')
    divisor = math.pi if diffuse_units == 'irradiance' else 1.
    direct = m * (rho * e / divisor + s) * k * exposure
    return {'proxy': a + direct, 'flash': direct}


def check_appearance_and_geometry(rgb, normals, face_bbox, *,
                                  min_luminance_span=.02, min_gradient=.003,
                                  min_normal_relief=.04, min_valid_fraction=.95):
    """ADVISORY face-ROI diagnostics; never certifies facial anatomy/identity.

    rgb linear BHWC; signed normals in the same convention as transport.
    Boxes (x0,y0,x1,y1), exclusive end, one per batch (or a shared box).
    Thresholds are uncalibrated starting values, not universal acceptance gates.
    RGB/normal gradient agreement is reported, NOT thresholded: eyebrows/iris
    edges need not be geometry edges. No landmarks are detected by this function.
    Even an 'review' result needs a landmark/semantic check before auto-routing.
    """
    c = _image(rgb, 'rgb', (3,))
    if (c < 0).any():
        raise ValueError('rgb must be nonnegative')
    n = _field(normals, c, 'normals', (3,), low=-1, high=1)
    thresholds = [_number(x, 'threshold') for x in (min_luminance_span, min_gradient, min_normal_relief, min_valid_fraction)]
    if min_valid_fraction > 1:
        raise ValueError('min_valid_fraction must be <=1')
    boxes = torch.as_tensor(face_bbox)
    if boxes.shape == (4,):
        boxes = boxes.unsqueeze(0).expand(c.shape[0], -1)
    if boxes.shape != (c.shape[0], 4) or not torch.isfinite(boxes).all() or (boxes != boxes.round()).any():
        raise ValueError('face_bbox must contain integer coordinates, shape (4,) or (B,4)')
    reports = []
    for i, box in enumerate(boxes.tolist()):
        x0, y0, x1, y1 = map(int, box)
        if not (0 <= x0 < x1 <= c.shape[2] and 0 <= y0 < y1 <= c.shape[1]) or min(x1-x0, y1-y0) < 8:
            raise ValueError('face box must be in bounds and at least 8x8')
        lum = _luma(c[i:i+1,y0:y1,x0:x1])
        normal = n[i:i+1,y0:y1,x0:x1]
        lengths = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        valid = lengths > 1e-6
        unit = normal / lengths.clamp_min(1e-8)
        # Blur before differencing to reduce single-pixel noise sensitivity.
        def smooth(x):
            return F.avg_pool2d(F.pad(x.permute(0,3,1,2), (1,1,1,1), mode='replicate'),3,1).permute(0,2,3,1)
        def gradient(x):
            x = smooth(x)
            gx = x[:,1:-1,2:] - x[:,1:-1,:-2]
            gy = x[:,2:,1:-1] - x[:,:-2,1:-1]
            return (gx.square().sum(-1) + gy.square().sum(-1)).sqrt() / 2
        g, gn = gradient(lum), gradient(unit)
        # Quantiles on CPU are also supported when inputs live on MPS.
        q = torch.quantile(lum.detach().float().cpu(), torch.tensor([.1,.9]))
        span, grad = float(q[1]-q[0]), float(g.mean())
        good = unit[valid.expand_as(unit)].reshape(-1,3)
        relief = float(good.std(0, unbiased=False).norm()) if len(good) else 0.
        alignment = float((g*gn).sum() / (g.norm()*gn.norm()).clamp_min(1e-8))
        fraction = float(valid.float().mean())
        reasons = []
        for failed, reason in ((span < thresholds[0], 'low_luminance_span'),
                               (grad < thresholds[1], 'low_image_gradient'),
                               (relief < thresholds[2], 'low_normal_relief'),
                               (fraction < thresholds[3], 'invalid_normals')):
            if failed: reasons.append(reason)
        reports.append({'status': 'needs_review_or_repair' if reasons else 'review',
                        'reasons': reasons, 'luminance_span': span, 'gradient': grad,
                        'normal_relief': relief, 'gradient_agreement': alignment,
                        'valid_normal_fraction': fraction, 'semantic_validation_required': True})
    return reports

# ---- Bridge layer: assumed geometry and independent ambient exposure ----
import json


SCENE_SCALE_PRESETS = {
    'Studio / Portrait': (0.3, 5.0),
    'Interior / Room': (0.5, 10.0),
    'Large Stage / Enclosure': (0.5, 20.0),
    'Exterior / Open Space': (0.5, 80.0),
}
SCENE_SCALE_ALIASES = {
    'cave': 'Large Stage / Enclosure',
    'interior': 'Interior / Room',
    'exterior': 'Exterior / Open Space',
    'studio': 'Studio / Portrait',
}


def auto_calibrate_transport_geometry(depth, subject_mask, fov_deg=50.,
                                      environment='Inside',
                                      ambient_lighting='In-Between',
                                      space_measurement='Meters (Depth)',
                                      space_size=6.0,
                                      subject_distance_m=2.0,
                                      *, near_m=None, far_m=None, scene_type=None,
                                      room_depth_m=None):
    """Assumed pinhole camera + bounded monotone relative-depth remapping.

    Artist-Centric Parameterization:
      - environment: 'Inside' (enclosed room/stage with boundary walls) or 'Outside' (open sky/exterior).
      - ambient_lighting: 'In-Between' (Balanced), 'Bright' (Day / High-Key), 'Dark' (Night / Low-Key).
      - space_measurement: 'Meters (Depth)', 'Square Meters (m²)', 'Cubic Meters (m³)'.
      - space_size: Numerical size of the space (depth in meters, area in m², or volume in m³).
      - subject_distance_m: Camera to subject distance in meters.
      - fov_deg: Horizontal camera field of view in degrees.
    """
    d = _image(depth, 'depth', (1,))
    if (d < 0).any() or (d > 1).any():
        raise ValueError('depth must be normalized to [0,1], near=1')
    if subject_mask is not None:
        m = _mask(subject_mask, d, 'subject_mask')
    else:
        split = float(d.median().item())
        m = torch.clamp((d - split + 0.05) / 0.10, 0.0, 1.0)

    # Legacy compatibility check: if environment or scene_type has a legacy preset name
    legacy_candidate = scene_type or (environment if isinstance(environment, str) and (environment in SCENE_SCALE_ALIASES or environment in SCENE_SCALE_PRESETS) else None)
    if legacy_candidate:
        resolved = SCENE_SCALE_ALIASES.get(legacy_candidate, legacy_candidate)
        preset_near, preset_far = SCENE_SCALE_PRESETS.get(resolved, (0.5, 20.0))
        is_outside = 'exterior' in str(legacy_candidate).lower() or 'open space' in str(legacy_candidate).lower()
        environment = 'Outside' if is_outside else 'Inside'
        effective_room_depth = preset_far
    else:
        preset_near = None
        preset_far = None
        val = float(room_depth_m if room_depth_m is not None else space_size)
        meas_str = str(space_measurement).lower()
        if 'square' in meas_str or 'm²' in meas_str or 'sqm' in meas_str:
            effective_room_depth = math.sqrt(max(0.1, val))
        elif 'cubic' in meas_str or 'm³' in meas_str or 'cbm' in meas_str:
            effective_room_depth = max(0.1, val) ** (1.0 / 3.0)
        else:
            effective_room_depth = val
        effective_room_depth = round(effective_room_depth, 4)

    is_inside = not str(environment).lower().startswith('out')

    fov = float(fov_deg)
    if not math.isfinite(fov) or not 5 <= fov <= 150:
        raise ValueError('horizontal fov_deg must be in [5,150]')

    if near_m is not None:
        near = _number(near_m, 'near_m', strictly=True)
    elif preset_near is not None:
        near = _number(preset_near, 'near_m', strictly=True)
    else:
        near = 0.3 if is_inside else 0.5

    subject = _number(subject_distance_m, 'subject_distance_m', strictly=True)

    if far_m is not None:
        far = _number(far_m, 'far_m', strictly=True)
    elif preset_far is not None:
        far = _number(preset_far, 'far_m', strictly=True)
    elif is_inside:
        far = max(subject + 0.5, effective_room_depth)
    else:
        far = 80.0

    if not near < subject < far:
        raise ValueError(f'require near_m ({near}m) < subject_distance_m ({subject}m) < far_m ({far}m)')

    ambient_str = str(ambient_lighting).lower()
    if 'bright' in ambient_str:
        ambient_ev = 0.5
    elif 'dark' in ambient_str:
        ambient_ev = -2.0
    else:
        ambient_ev = 0.0

    h, w = d.shape[1:3]
    focal = w / (2 * math.tan(math.radians(fov) / 2))
    target = (1/subject - 1/far) / (1/near - 1/far)
    warped, diagnostics = [], []
    for i in range(d.shape[0]):
        di, mi = d[i:i+1], m[i:i+1]
        selected = di[mi > .5]
        span = float(di.max()-di.min())
        warnings = ['metric_scale_and_fov_are_assumptions', 'receiver_mask_required_to_exclude_sky_air']
        median, k = None, 1.
        if selected.numel() < 4:
            warnings.append('insufficient_subject_support_using_unwarped_depth')
        elif span < 1e-4:
            warnings.append('constant_depth_no_geometric_relief')
        else:
            median = float(selected.median())
            if not 1e-4 < median < 1-1e-4:
                warnings.append('subject_depth_at_endpoint_cannot_fit_distance')
            else:
                # t=d/(d+k*(1-d)); fixes 0 and 1; positive smooth derivative.
                raw_k = median*(1-target)/(target*(1-median))
                k = min(100., max(.01, raw_k))
                if k != raw_k:
                    warnings.append('depth_warp_limited_subject_anchor_approximate')
        t = di / (di + k*(1-di))
        warped.append(t)
        actual_z = None if median is None else 1/((1-median/(median+k*(1-median)))/far + (median/(median+k*(1-median)))/near)
        diagnostics.append({'warnings': warnings, 'warp_k': k, 'subject_relative_median': median,
                            'mapped_subject_distance_m': actual_z, 'relative_span': span,
                            'geometry_has_relief': span >= 1e-4})
    warped = torch.cat(warped)
    z = ((1-warped)/far + warped/near).reciprocal()
    return {'schema': 1, 'warped_depth': warped, 'raw_depth': d, 'subject_mask': m, 'depth_m': z,
            'intrinsics': (focal, focal, (w-1)/2, (h-1)/2),
            'depth_anchors': ((0., far), (1., near)), 'depth_model': 'inverse',
            'coordinate_convention': 'x_right_y_down_z_forward',
            'scale_source': 'assumed_scene_prior',
            'environment': environment,
            'ambient_lighting': ambient_lighting,
            'ambient_ev': ambient_ev,
            'effective_room_depth_m': effective_room_depth if is_inside else far,
            'scene_type': scene_type or f"{environment} ({ambient_lighting})",
            'diagnostics': diagnostics}


def derive_ambient_baseline(readable_draft_rgb, ambient_tint_field, target_exposure_ev,
                            *, tint_strength=1., emission_mask=None):
    """Artistic ambient transfer in linear RGB, not inverse rendering.

    A=2**EV * draft * ((1-s)+s*tint/Y(tint)); black tint means neutral.
    Tint is a supplied LIGHT COLOUR field, never sampled automatically from a
    laser/background image. It is per-pixel luminance normalized; input tint
    brightness does not set ambient exposure. Positive channel floors are NOT
    invented. At saturated tints a channel may vanish in this ambient branch;
    draft/prior and subsequent neutral strobe remain independent and unchanged.
    Optional emission mask preserves draft emitters instead of dimming them.
    Returns HDR linear output without clamp. EV is relative to the draft.
    """
    c = _image(readable_draft_rgb, 'readable_draft_rgb', (3,))
    if (c < 0).any():
        raise ValueError('readable_draft_rgb must be nonnegative')
    tint = _field(ambient_tint_field, c, 'ambient_tint_field', (3,))
    ev = float(target_exposure_ev)
    if not math.isfinite(ev) or not -24 <= ev <= 16:
        raise ValueError('target_exposure_ev must be in [-24,16]')
    strength = _number(tint_strength, 'tint_strength')
    if strength > 1:
        raise ValueError('tint_strength must be <=1')
    y = _luma(tint)
    chroma = torch.where(y > 1e-8, tint / y.clamp_min(1e-8), torch.ones_like(tint))
    transfer = (2.**ev) * ((1-strength) + strength*chroma)
    ambient = c * transfer
    if emission_mask is not None:
        mask = _field(emission_mask, c, 'emission_mask', (1,), high=1)
        ambient = ambient*(1-mask) + c*mask
    return {'ambient': ambient, 'transfer': transfer,
            'black_tint_fallback': y <= 1e-8}


def srgb_to_linear(image):
    x = _image(image, 'sRGB image', (3,))
    if (x < 0).any() or (x > 1).any():
        raise ValueError('sRGB IMAGE must be in [0,1]')
    return torch.where(x <= .04045, x/12.92, ((x+.055)/1.055).pow(2.4))


def linear_to_srgb(image):
    x = _image(image, 'linear image', (3,))
    if (x < 0).any():
        raise ValueError('linear image must be nonnegative')
    return torch.where(x <= .0031308, x*12.92, 1.055*x.pow(1/2.4)-.055)


def _preview(linear, mode='clip'):
    if mode == 'reinhard':
        linear = linear/(1+_luma(linear))
    elif mode != 'clip':
        raise ValueError('preview mode must be clip or reinhard')
    return linear_to_srgb(linear).clamp(0,1)


def _mask(mask, ref, name='mask'):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(-1)
        elif mask.ndim == 4 and mask.shape[-1] in (3, 4):
            mask = mask[..., :3].mean(dim=-1, keepdim=True)
        elif mask.ndim == 4 and mask.shape[1] == 1 and mask.shape[-1] != 1:
            mask = mask.permute(0, 2, 3, 1)
        if mask.device != ref.device:
            mask = mask.to(device=ref.device)
        if mask.shape[1:3] != ref.shape[1:3]:
            m_t = mask.permute(0, 3, 1, 2)
            m_t = F.interpolate(m_t, size=ref.shape[1:3], mode='bilinear', align_corners=False)
            mask = m_t.permute(0, 2, 3, 1)
    return _field(mask, ref, name, (1,), high=1)


def _normals(image, convention):
    x = _image(image, 'normal image', (3,))
    if (x < 0).any() or (x > 1).any():
        raise ValueError('normal IMAGE must be encoded in [0,1]')
    # Normal images are data: NEVER apply sRGB decoding.
    n = x*2-1
    if convention == 'encoded_x_right_y_up_z_toward_camera':
        n = n*n.new_tensor([1,-1,-1])
    elif convention != 'encoded_x_right_y_down_z_forward':
        raise ValueError('unknown normal convention')
    return n


def _linear_payload(value):
    return {'schema': 1, 'rgb': value}


def _read_linear(value):
    if not isinstance(value, dict) or value.get('schema') != 1 or 'rgb' not in value:
        raise ValueError('expected SL_LINEAR schema 1')
    return _image(value['rgb'], 'SL_LINEAR.rgb', (3,))


def _heatmap(e):
    y = _luma(e)
    t = y/(1+y)  # fixed scale across frames, not per-image normalization
    return torch.cat((t, 4*t*(1-t), 1-t), -1).clamp(0,1)


_NORMAL_OPTIONS = ['encoded_x_right_y_down_z_forward', 'encoded_x_right_y_up_z_toward_camera']


class StudioTransportGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'depth': ('IMAGE',),
                'environment': (['Inside', 'Outside'], {'default': 'Inside'}),
                'ambient_lighting': (['In-Between', 'Bright', 'Dark'], {'default': 'In-Between'}),
                'space_measurement': (['Meters (Depth)', 'Square Meters (m²)', 'Cubic Meters (m³)'], {'default': 'Meters (Depth)'}),
                'space_size': ('FLOAT', {'default': 6.0, 'min': 0.5, 'max': 50000.0, 'step': 0.5}),
                'subject_distance_m': ('FLOAT', {'default': 2.0, 'min': 0.1, 'max': 50.0, 'step': 0.1}),
                'fov_deg': ('FLOAT', {'default': 50.0, 'min': 5.0, 'max': 150.0, 'step': 1.0}),
            },
            'optional': {
                'subject_mask': ('MASK',),
                'near_m_override': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10000.0, 'step': 0.1}),
                'far_m_override': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 10000.0, 'step': 0.1}),
            }
        }
    RETURN_TYPES = ('SL_GEOMETRY', 'IMAGE', 'STRING')
    RETURN_NAMES = ('geometry', 'assumed_depth_preview', 'report')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self, depth, subject_mask=None, environment='Inside', ambient_lighting='In-Between',
            space_measurement='Meters (Depth)', space_size=6.0, subject_distance_m=2.0, fov_deg=50.0,
            near_m_override=0.0, far_m_override=0.0, **kwargs):
        d = _image(depth, 'depth', (1, 3))
        if d.shape[-1] == 3 and not torch.allclose(d, d[..., :1].expand_as(d), atol=1e-5, rtol=0):
            raise ValueError('depth IMAGE must be grayscale data, not a colour visualization')
        d = d[..., :1]
        near = _number(near_m_override, 'near_m_override')
        far = _number(far_m_override, 'far_m_override')

        # Handle legacy positional calls: (depth, subject_mask, fov_deg, scene_type, ...)
        if isinstance(environment, (int, float)):
            fov_deg = float(environment)
            environment = str(ambient_lighting) if isinstance(ambient_lighting, str) else 'Inside'
            ambient_lighting = 'In-Between'

        scene_type = kwargs.get('scene_type', None)
        room_depth_m = kwargs.get('room_depth_m', None)

        m_sub = _mask(subject_mask, d, 'subject_mask') if subject_mask is not None else None

        result = auto_calibrate_transport_geometry(
            d, m_sub, fov_deg=fov_deg,
            environment=environment,
            ambient_lighting=ambient_lighting,
            space_measurement=space_measurement,
            space_size=space_size,
            subject_distance_m=subject_distance_m,
            near_m=near or None,
            far_m=far or None,
            scene_type=scene_type,
            room_depth_m=room_depth_m
        )
        report = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
        return result, result['warped_depth'].expand(-1, -1, -1, 3).clone(), json.dumps(report)


class StudioTransportAmbient:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'readable_draft': ('IMAGE',), 'ambient_tint': ('IMAGE',),
                'target_exposure_ev': ('FLOAT', {'default':-3.,'min':-24.,'max':16.}),
                'tint_strength': ('FLOAT', {'default':1.,'min':0.,'max':1.})},
                'optional': {'emission_mask': ('MASK',)}}
    RETURN_TYPES = ('SL_LINEAR','SL_LINEAR','IMAGE')
    RETURN_NAMES = ('ambient_linear','draft_linear','ambient_preview')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self, readable_draft, ambient_tint, target_exposure_ev=-3., tint_strength=1., emission_mask=None):
        c = srgb_to_linear(readable_draft)
        mask = None if emission_mask is None else _mask(emission_mask,c)
        result = derive_ambient_baseline(c,srgb_to_linear(ambient_tint),target_exposure_ev,
                                       tint_strength=tint_strength,emission_mask=mask)
        return _linear_payload(result['ambient']),_linear_payload(c),_preview(result['ambient'])


class StudioTransportReflectance:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'draft_linear': ('SL_LINEAR',), 'prior_plate': ('IMAGE',), 'mask': ('MASK',),
                'field_is_calibrated': ('BOOLEAN', {'default':False})},
                'optional': {'ambient_transfer': ('SL_LINEAR',)}}
    RETURN_TYPES = ('SL_LINEAR','IMAGE','MASK')
    RETURN_NAMES = ('reflectance_linear','reflectance_preview','unresolved_mask')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self, draft_linear, prior_plate, mask, field_is_calibrated=False, ambient_transfer=None):
        c = _read_linear(draft_linear)
        if field_is_calibrated and ambient_transfer is None:
            raise ValueError('calibrated inversion requires ambient_transfer=E_ambient/pi for the ORIGINAL draft')
        result = estimate_surface_reflectance(c,None if ambient_transfer is None else _read_linear(ambient_transfer),
                    _mask(mask,c),prior=srgb_to_linear(prior_plate),field_is_calibrated=field_is_calibrated)
        return _linear_payload(result['reflectance']),_preview(result['reflectance']),result['unresolved'][...,0].float()


class StudioTransportWall:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'geometry': ('SL_GEOMETRY',),
                'normals': ('IMAGE',),
                'normal_convention': (_NORMAL_OPTIONS,),
                'light_x': ('FLOAT', {'default': 0.}),
                'light_y': ('FLOAT', {'default': 0.}),
                'light_z': ('FLOAT', {'default': 0.}),
                'intensity_r': ('FLOAT', {'default': 1., 'min': 0.}),
                'intensity_g': ('FLOAT', {'default': 1., 'min': 0.}),
                'intensity_b': ('FLOAT', {'default': 1., 'min': 0.}),
                'min_distance_m': ('FLOAT', {'default': 1.0, 'min': .001}),
            },
            'optional': {
                'receiver_mask': ('MASK',),
                'visibility': ('MASK',),
                'telemetry': ('STUDIOLIGHT_TELEMETRY',),
                'rig': ('SL_RIG',),
            }
        }

    RETURN_TYPES = ('SL_FLASH', 'IMAGE', 'MASK')
    RETURN_NAMES = ('flash_transport', 'irradiance_heatmap', 'valid_geometry')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self, geometry, normals, normal_convention, receiver_mask=None, visibility=None,
            light_x=0., light_y=0., light_z=0., intensity_r=1., intensity_g=1., intensity_b=1.,
            min_distance_m=1.0, telemetry=None, rig=None):
        if not isinstance(geometry, dict) or geometry.get('schema') != 1 or geometry.get('coordinate_convention') != 'x_right_y_down_z_forward':
            raise ValueError('expected SL_GEOMETRY schema 1 in camera coordinates')
        d = geometry['warped_depth']
        raw_d = geometry.get('raw_depth', d)

        # Determine receiver mask (auto-derived background receiver if empty/zero)
        m_rec = None
        if receiver_mask is not None and isinstance(receiver_mask, torch.Tensor) and receiver_mask.numel() > 0:
            m_test = _mask(receiver_mask, d)
            if float(m_test.max().item()) > 0.05:
                m_rec = m_test
        if m_rec is None:
            if isinstance(telemetry, dict) and telemetry.get('fg_confidence') is not None:
                m_fg = _mask(telemetry['fg_confidence'], d)
            elif geometry.get('subject_mask') is not None:
                m_fg = _mask(geometry['subject_mask'], d)
            else:
                split = float(d.median().item())
                m_fg = torch.clamp((d - split + 0.05) / 0.10, 0.0, 1.0)
            # Dilate foreground mask by 9px so wall transport never touches dark hair crown or silhouette edges
            m_fg_dilated = F.max_pool2d(
                F.pad(m_fg.permute(0, 3, 1, 2), (4, 4, 4, 4), mode='replicate'),
                kernel_size=9, stride=1
            ).permute(0, 2, 3, 1)
            # Distance proximity gate: distant street/sky (raw_d < 0.22) smoothly gates to 0.0
            prox_gate = torch.clamp((raw_d - 0.22) / 0.16, 0.0, 1.0)
            prox_gate = prox_gate * prox_gate * (3.0 - 2.0 * prox_gate)
            m_rec = ((1.0 - m_fg_dilated) * prox_gate).clamp(0.0, 1.0)

        # Determine visibility mask (auto-derived from telemetry or depth if empty/zero)
        m_vis = None
        if visibility is not None and isinstance(visibility, torch.Tensor) and visibility.numel() > 0:
            v_test = _mask(visibility, d)
            if float(v_test.max().item()) > 0.05:
                m_vis = v_test
        if m_vis is None:
            if isinstance(telemetry, dict) and 'shadow_occlusion' in telemetry:
                occ = _mask(telemetry['shadow_occlusion'], d)
                m_vis = (1.0 - occ).clamp(0.0, 1.0)
            else:
                m_vis = torch.ones_like(d)

        # Check for unified 3-light rig state
        lights = None
        if isinstance(telemetry, dict) and 'lights' in telemetry:
            lights = telemetry['lights']
        elif isinstance(rig, dict) and 'lights' in rig:
            lights = rig['lights']

        norm_tensor = _normals(normals, normal_convention)
        intrinsics = geometry['intrinsics']
        anchors = geometry['depth_anchors']

        if lights and isinstance(lights, list) and len(lights) > 0:
            # Multi-fixture direct cavity wall transport matching the exact StudioLightRig console
            total_irradiance = torch.zeros((d.shape[0], d.shape[1], d.shape[2], 3), device=d.device, dtype=torch.float32)
            valid_mask = torch.ones_like(d[..., 0])
            eff_rmin = max(float(min_distance_m), 1.2)
            for light in lights:
                if not light.get('enabled', True) or light.get('intensity', 0) <= 0:
                    continue
                az = math.radians(float(light.get('azimuth', 0.0)))
                el = math.radians(float(light.get('elevation', 0.0)))
                dist = max(0.5, float(light.get('distance', 1.8)))
                # Direction in camera coordinates (+X right, +Y down, +Z forward)
                lx = dist * math.sin(az) * math.cos(el)
                ly = -dist * math.sin(el)
                lz = dist * math.cos(az) * math.cos(el) + float(geometry.get('effective_room_depth_m', 2.0)) * 0.25

                k_rgb = kelvin_to_rgb(int(light.get('kelvin', 5500)))
                # Calibrated linear irradiance scale (no 10.0x nuclear multiplier)
                inten = float(light.get('intensity', 1.0)) * 0.85
                ir, ig, ib = k_rgb[0] * inten, k_rgb[1] * inten, k_rgb[2] * inten

                res = compute_cavity_wall_transport(
                    d, norm_tensor, (lx, ly, lz), (ir, ig, ib),
                    intrinsics=intrinsics, depth_anchors=anchors,
                    receiver_mask=m_rec, visibility=m_vis, min_distance_m=eff_rmin
                )
                total_irradiance += res['irradiance']
                valid_mask = valid_mask * res['valid'][..., 0].float()

            e = total_irradiance.clamp(0.0, 1.5)
            payload = {
                'schema': 1, 'diffuse': e, 'specular': torch.zeros_like(e),
                'units': 'irradiance', 'receiver_mask_applied': True,
                'receiver_mask': m_rec, 'visibility': m_vis
            }
            return payload, _heatmap(e), valid_mask
        else:
            result = compute_cavity_wall_transport(
                d, norm_tensor, (light_x, light_y, light_z), (intensity_r, intensity_g, intensity_b),
                intrinsics=intrinsics, depth_anchors=anchors,
                receiver_mask=m_rec, visibility=m_vis, min_distance_m=min_distance_m
            )
            e = result['irradiance']
            payload = {
                'schema': 1, 'diffuse': e, 'specular': torch.zeros_like(e),
                'units': 'irradiance', 'receiver_mask_applied': True,
                'receiver_mask': m_rec, 'visibility': m_vis
            }
            return payload, _heatmap(e), result['valid'][..., 0].float()


class StudioTransportPreflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'draft': ('IMAGE',),
                'normals': ('IMAGE',),
                'normal_convention': (_NORMAL_OPTIONS,),
                'x0': ('INT', {'default': 0, 'min': 0}),
                'y0': ('INT', {'default': 0, 'min': 0}),
                'x1': ('INT', {'default': 0, 'min': 0}),
                'y1': ('INT', {'default': 0, 'min': 0}),
            },
            'optional': {
                'depth_optional': ('IMAGE',),
                'subject_mask_optional': ('MASK',),
            }
        }

    RETURN_TYPES = ('IMAGE', 'STRING')
    RETURN_NAMES = ('review_overlay', 'review_report')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self, draft, normals, normal_convention, x0=0, y0=0, x1=0, y1=0,
            depth_optional=None, subject_mask_optional=None):
        c = srgb_to_linear(draft)
        b, h, w, _ = c.shape

        # Bug 4: Confine Face Preflight to the Subject Head Region.
        # Auto-detect upper-body / head ROI from depth / foreground mask when x0==y0==x1==y1==0
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            if depth_optional is not None:
                d = _image(depth_optional, 'depth_optional', (1, 3))[..., :1]
                split = float(d.median().item())
                fg = (d > split).float()
                upper_fg = fg[:, :int(0.6 * h), :, 0]
                coords = torch.nonzero(upper_fg[0] > 0.5)
                if len(coords) > 64:
                    y_min, x_min = coords.min(0).values.tolist()
                    y_max, x_max = coords.max(0).values.tolist()
                    pad_x = int((x_max - x_min) * 0.10)
                    pad_y = int((y_max - y_min) * 0.10)
                    x0 = max(0, x_min - pad_x)
                    x1 = min(w, x_max + pad_x)
                    y0 = max(0, y_min - pad_y)
                    y1 = min(h, y_max + pad_y)
            elif subject_mask_optional is not None:
                m = _mask(subject_mask_optional, c)
                upper_m = m[:, :int(0.6 * h), :, 0]
                coords = torch.nonzero(upper_m[0] > 0.5)
                if len(coords) > 64:
                    y_min, x_min = coords.min(0).values.tolist()
                    y_max, x_max = coords.max(0).values.tolist()
                    x0, y0 = max(0, x_min), max(0, y_min)
                    x1, y1 = min(w, x_max), min(h, y_max)

        # Fallback to centered upper third if not detected or empty
        if x0 >= x1 or y0 >= y1 or (x1 - x0) < 16 or (y1 - y0) < 16:
            x0 = int(0.25 * w)
            x1 = int(0.75 * w)
            y0 = int(0.08 * h)
            y1 = int(0.52 * h)

        reports = check_appearance_and_geometry(c, _normals(normals, normal_convention), (x0, y0, x1, y1))
        overlay = draft.clone()
        for i, report in enumerate(reports):
            report['roi'] = [x0, y0, x1, y1]
            report['roi_must_be_face'] = True
            color = overlay.new_tensor([1., .1, .1] if report['reasons'] else [1., .7, 0.])
            overlay[i, y0:y1, x0:x1] = .85 * overlay[i, y0:y1, x0:x1] + .15 * color
            overlay[i, y0:y0+1, x0:x1] = color
            overlay[i, y1-1:y1, x0:x1] = color
            overlay[i, y0:y1, x0:x0+1] = color
            overlay[i, y0:y1, x1-1:x1] = color
        return overlay, json.dumps(reports)


class StudioTransportComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'ambient_linear': ('SL_LINEAR',),'reflectance_linear': ('SL_LINEAR',),
                'flash_transport': ('SL_FLASH',),
                'strength': ('FLOAT',{'default':1.,'min':0.}),
                'exposure_scale': ('FLOAT',{'default':1.,'min':0.}),
                'display_mapping': (['clip','reinhard'],)}}
    RETURN_TYPES = ('SL_LINEAR','IMAGE','IMAGE')
    RETURN_NAMES = ('proxy_linear','proxy_image','flash_heatmap')
    FUNCTION = 'run'
    CATEGORY = 'StudioLight/Transport'

    def run(self,ambient_linear,reflectance_linear,flash_transport,strength=1.,exposure_scale=1.,display_mapping='clip'):
        a,rho = _read_linear(ambient_linear),_read_linear(reflectance_linear)
        p = flash_transport
        if not isinstance(p,dict) or p.get('schema') != 1 or p.get('receiver_mask_applied') is not True:
            raise ValueError('SL_FLASH must be schema 1 with receiver mask already applied')
        result = additive_flash_proxy(a,rho,p['diffuse'],p['specular'],torch.ones_like(a[...,:1]),
                    diffuse_units=p['units'],strength=strength,exposure_scale=exposure_scale)
        return _linear_payload(result['proxy']),_preview(result['proxy'],display_mapping),_heatmap(result['flash'])


NODE_CLASS_MAPPINGS = {
    'StudioTransportGeometry': StudioTransportGeometry,
    'StudioTransportAmbient': StudioTransportAmbient,
    'StudioTransportReflectance': StudioTransportReflectance,
    'StudioTransportWall': StudioTransportWall,
    'StudioTransportPreflight': StudioTransportPreflight,
    'StudioTransportComposite': StudioTransportComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    'StudioTransportGeometry': 'StudioLight · Assumed Geometry',
    'StudioTransportAmbient': 'StudioLight · Ambient Baseline',
    'StudioTransportReflectance': 'StudioLight · Reflectance Prior',
    'StudioTransportWall': 'StudioLight · Direct Receiver Transport',
    'StudioTransportPreflight': 'StudioLight · Preflight Review',
    'StudioTransportComposite': 'StudioLight · Additive Flash',
}
