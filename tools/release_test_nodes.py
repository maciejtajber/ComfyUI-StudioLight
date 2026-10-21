"""Test-only nodes. Installed separately by prepare_pod_release_tests.py; never production-registered."""
from pathlib import Path
import json
import numpy as np
import torch
import folder_paths

class StudioLightTestClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',)}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='clamp'
    CATEGORY='_StudioLightReleaseTests'
    def clamp(self,image): return (image.clamp(0,1),)

class StudioLightTestCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'prefix': ('STRING', {'default':'StudioLightRelease/test'})}}
    RETURN_TYPES=()
    FUNCTION='capture'
    OUTPUT_NODE=True
    CATEGORY='_StudioLightReleaseTests'
    def capture(self,image,prefix):
        root=Path(folder_paths.get_output_directory()).resolve()
        target=(root/(prefix+'.npz')).resolve()
        if root not in target.parents: raise ValueError('Capture prefix must stay under output directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        a=image.detach().float().cpu().numpy()
        finite=np.isfinite(a)
        if not finite.all(): raise ValueError('Nonfinite release-test tensor')
        np.savez_compressed(target,image=a)
        metrics={'shape':list(a.shape),'min':float(a.min()),'max':float(a.max()),'mean':float(a.mean()),
                 'p99':float(np.quantile(a,.99)),'channels_over_one_fraction':float((a>1).mean()),
                 'display_saturated_pixel_fraction':float((a.max(axis=-1)>=.99).mean())}
        meta=target.with_suffix('.json');meta.write_text(json.dumps(metrics,indent=2))
        def entry(path): return {'filename':path.name,'subfolder':str(path.parent.relative_to(root)),'type':'output'}
        return {'ui':{'studiolight_files':[entry(target),entry(meta)]}}

NODE_CLASS_MAPPINGS={'StudioLightTestClamp':StudioLightTestClamp,'StudioLightTestCapture':StudioLightTestCapture}


class StudioLightTestLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'file': ('STRING', {'default': ''})}}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='load'
    CATEGORY='_StudioLightReleaseTests'
    @classmethod
    def IS_CHANGED(cls,file):
        import hashlib
        root=Path(folder_paths.get_output_directory()).resolve()
        target=(root/file).resolve()
        if root not in target.parents or not target.is_file(): raise ValueError('Replay file must be inside the output directory')
        return hashlib.sha256(target.read_bytes()).hexdigest()
    def load(self,file):
        root=Path(folder_paths.get_output_directory()).resolve()
        target=(root/file).resolve()
        if root not in target.parents or target.suffix!='.npz': raise ValueError('Replay file must be an NPZ inside the output directory')
        with np.load(target,allow_pickle=False) as data:
            a=data['image']
            if a.ndim!=4 or a.shape[-1] not in (1,3) or not np.isfinite(a).all(): raise ValueError('Invalid saved replay tensor')
            return (torch.from_numpy(a.astype(np.float32,copy=True)),)

NODE_CLASS_MAPPINGS['StudioLightTestLoad']=StudioLightTestLoad


class StudioLightTestClampTelemetry:
    """Clamp separated HDR gains too, so telemetry cannot bypass the experiment."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'telemetry': ('STUDIOLIGHT_TELEMETRY',)}}
    RETURN_TYPES = ('STUDIOLIGHT_TELEMETRY',)
    FUNCTION = 'clamp'
    CATEGORY = '_StudioLightReleaseTests'
    def clamp(self, telemetry):
        return ({key: value.clamp(0, 1) if key in ('direct_diffuse', 'direct_specular') else value
                 for key, value in telemetry.items()},)

NODE_CLASS_MAPPINGS['StudioLightTestClampTelemetry'] = StudioLightTestClampTelemetry
