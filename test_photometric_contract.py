"""Regression gates for separated light transport, safe priors and diagnostics."""
import json
import os
from pathlib import Path
import unittest
import torch
from nodes.photometric_composite import StudioPhotometricComposite, srgb_to_linear, linear_to_srgb, rgb_to_luminance
from nodes.photometric_support import enclosure_bounce, recover_masked_shadows
from nodes.shader_engine import render_studio_lighting, reconstruct_metric_surface
from nodes.studio_light import StudioLightRig


class PhotometricContractTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get('STUDIOLIGHT_TEST_DEVICE', 'cpu'))
        self.node = StudioPhotometricComposite()
        self.image = torch.full((1, 24, 24, 3), .2, device=self.device)
        self.light = torch.ones_like(self.image)
        self.fg = torch.zeros_like(self.image[..., :1]); self.fg[:, 6:18, 6:18] = 1
        self.depth = .3 + self.fg * .25
        self.tele = dict(d_norm=self.depth, fg_confidence=self.fg, d_split=.35,
                         direct_diffuse=self.light, direct_specular=torch.zeros_like(self.light),
                         ambient_rgb=torch.zeros_like(self.light),
                         shadow_occlusion=(1-self.fg)*.8, volumetric_beam=self.light*.4)

    def compose(self, **kw):
        args=dict(image=self.image, lighting_map=self.light, telemetry=self.tele,
                  lighting_strength=1., specular_kick=0., crevice_delight=0.)
        args.update(kw)
        return self.node.composite(**args)

    def test_zero_strobe_has_zero_foreground_and_preserves_background(self):
        result,sheet=self.compose(lighting_strength=0., room_bounce_strength=1., atmospheric_haze=1.)
        self.assertEqual(float(result[:,8:16,8:16].max()),0.)
        self.assertTrue(torch.allclose(result[:,:3], self.image[:,:3], atol=1e-6))
        # Diagnostic red/green channels below the second header are dark.
        h,w=12,12
        diag=sheet[:,(h+20)+20:,w:]
        self.assertEqual(float(diag[...,:2].max()),0.)

    def test_zero_strobe_cyclorama_restores_background(self):
        tele={**self.tele, 'd_norm':self.depth*.0+.30, 'd_split':.35,
              'fg_confidence':torch.zeros_like(self.fg)}
        result,_=self.compose(telemetry=tele,lighting_strength=0.,atmospheric_haze=1.)
        self.assertTrue(torch.allclose(result,self.image,atol=2e-6))

    def test_coloured_practical_survives_background_misclassification(self):
        image=self.image.clone();image[:,10:14,10:14]=torch.tensor([1.,.25,.01],device=self.device)
        tele={**self.tele, 'fg_confidence':torch.zeros_like(self.fg),
              'd_norm':torch.full_like(self.fg,.2), 'shadow_occlusion':torch.ones_like(self.fg)}
        result,_=self.compose(image=image,telemetry=tele,lighting_strength=1.,wall_shadow_density=.95)
        # Protection occurs before the intentionally compressive highlight shoulder.
        from nodes.photometric_composite import filmic_reinhard_tonemap
        expected=linear_to_srgb(filmic_reinhard_tonemap(srgb_to_linear(image)))
        self.assertTrue(torch.allclose(result[:,10:14,10:14],expected[:,10:14,10:14],atol=1e-6))
        unprotected=self.compose(image=image,telemetry=tele,lighting_strength=1.,wall_shadow_density=.95,
                                 emission_mask=torch.zeros_like(self.fg))[0]
        self.assertLess(float(unprotected[:,10:14,10:14].mean()),float(result[:,10:14,10:14].mean()))

    def test_rig_ambient_is_independent_of_strobe_strength(self):
        tele={k:v for k,v in self.tele.items() if k not in ('d_norm','fg_confidence','shadow_occlusion','volumetric_beam')}
        tele['ambient_rgb']=self.light*.05;tele['direct_diffuse']=torch.zeros_like(self.light)
        a=self.compose(telemetry=tele,lighting_strength=0.)[0]
        b=self.compose(telemetry=tele,lighting_strength=2.)[0]
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.allclose(srgb_to_linear(a),srgb_to_linear(self.image)*.05,atol=1e-7))

    def test_separable_average_matches_high_precision_box_reference(self):
        import torch.nn.functional as F
        from nodes.photometric_composite import _box_filter_bchw
        x=torch.rand((1,3,17,21),generator=torch.Generator().manual_seed(71))
        # Independent double-precision 2-D reference checks border replication.
        expected=F.avg_pool2d(F.pad(x.double(),(6,6,6,6),mode='replicate'),13,1)
        actual=_box_filter_bchw(x.to(self.device),6).cpu().double()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=0))

    def test_bounce_is_additive_power_linear_and_coloured(self):
        field=self.light*torch.tensor([1.,.5,.25],device=self.device)
        def bounce(power, reflectance=.5):
            return enclosure_bounce(self.depth,self.fg,torch.zeros_like(self.fg),field*power,.22,reflectance)[0]
        a,b=bounce(1),bounce(2)
        self.assertGreater(float(a.max()),0)
        self.assertTrue(torch.allclose(b,a*2,atol=1e-7))
        self.assertTrue(torch.allclose(a[...,0],a[...,2]*4,atol=1e-7))
        self.assertEqual(float(bounce(0).max()),0.)
        self.assertEqual(float(bounce(1,0).max()),0.)

    def test_bounce_excludes_subject_illumination(self):
        field=self.light*self.fg
        bounce,_,_=enclosure_bounce(self.depth,self.fg,torch.zeros_like(self.fg),field,1.,1.)
        self.assertEqual(float(bounce.max()),0.)

    def test_unknown_constant_depth_missing_walls_and_open_override_have_no_bounce(self):
        for d,fg,field,override in [(None,self.fg,self.light,-1),
                                   (self.depth*0+.5,self.fg,self.light,-1),
                                   (self.depth,torch.ones_like(self.fg),self.light,-1),
                                   (self.depth,self.fg,None,-1),
                                   (self.depth,self.fg,self.light,0)]:
            with self.subTest(override=override):
                b,_,_=enclosure_bounce(d,fg,torch.zeros_like(self.fg),field,1.,1.,override)
                self.assertEqual(float(b.max()),0.)

    def test_bounce_does_not_reduce_hdr_foreground(self):
        for power in (0.,.5,1.,2.):
            no=self.compose(lighting_strength=power,room_bounce_strength=0.,chroma_bleed=0.)[0]
            yes=self.compose(lighting_strength=power,room_bounce_strength=.5,chroma_bleed=0.)[0]
            self.assertTrue((yes[:,8:16,8:16]>=no[:,8:16,8:16]-1e-7).all())

    def test_black_diffuse_stays_black_without_reconstruction(self):
        result,_=self.compose(image=torch.zeros_like(self.image),room_bounce_strength=1.)
        self.assertEqual(float(result.max()),0.)

    def test_reconstruction_requires_mask_and_exposed_reference(self):
        with self.assertRaisesRegex(ValueError,'recovery_mask'):
            self.compose(shadow_recovery=.5)
        black=torch.zeros_like(self.image)
        repaired=recover_masked_shadows(black,self.fg,1.,rgb_to_luminance)
        self.assertTrue(torch.equal(black,repaired))

    def test_reconstruction_is_masked_material_coloured_and_light_independent(self):
        base=torch.zeros_like(self.image);base[:,8:12,8:12]=torch.tensor([.02,.04,.08],device=self.device)
        repaired=recover_masked_shadows(base,self.fg,1.,rgb_to_luminance)
        self.assertTrue(torch.equal(repaired*(1-self.fg),base*(1-self.fg)))
        self.assertGreater(float(repaired[0,15,15,2]),float(repaired[0,15,15,0]))
        self.assertLessEqual(float(rgb_to_luminance(repaired).max()),.05+1e-6)

    def test_heterogeneous_batches_match_solo_with_distinct_splits(self):
        tele2={**self.tele,'d_norm':self.depth*.8,'d_split':.27,'fg_confidence':self.fg.roll(4,2)}
        a=self.compose(room_bounce_strength=.3)
        b=self.compose(image=self.image*.5,telemetry=tele2,room_bounce_strength=.3)
        batched={k:torch.cat((v,tele2[k])) if isinstance(v,torch.Tensor) else torch.tensor([v,tele2[k]],device=self.device) for k,v in self.tele.items()}
        together=self.compose(image=torch.cat((self.image,self.image*.5)),telemetry=batched,room_bounce_strength=.3)
        for i in (0,1):
            self.assertTrue(torch.allclose(together[i],torch.cat((a[i],b[i])),atol=1e-6))

    def test_singleton_images_and_telemetry_broadcast_through_diagnostics(self):
        out,sheet=self.compose(lighting_map=self.light.expand(2,-1,-1,-1))
        self.assertEqual(out.shape[0],2);self.assertEqual(sheet.shape[0],2)
        self.assertTrue(torch.equal(sheet[0],sheet[1]))

    def test_explicit_telemetry_survives_image_clone(self):
        self.light._studiolight_telemetry=self.tele
        implicit=self.node.composite(self.image,self.light,specular_kick=0.)
        explicit=self.node.composite(self.image,self.light.clone(),telemetry=self.tele,specular_kick=0.)
        for a,b in zip(implicit,explicit):self.assertTrue(torch.equal(a,b))

    def test_tiny_odd_images_and_draft_gamma(self):
        for h,w in ((1,1),(1,7),(7,1),(5,9)):
            base=torch.full((1,h,w,3),.5,device=self.device)
            out,sheet=self.node.composite(base,torch.ones_like(base),specular_kick=0.)
            self.assertEqual(out.shape,base.shape)
            self.assertTrue(torch.isfinite(sheet).all())
            self.assertTrue(torch.allclose(sheet[:,20:20+max(1,h//2),:max(1,w//2)],torch.tensor(.5,device=self.device)))

    def test_invalid_shapes_nonfinite_and_batch_fail_early(self):
        for name,value in [('lighting_map',self.light*float('nan')),('image',self.image*float('inf')),
                           ('depth_map',torch.zeros(0,2,2,1,device=self.device)),
                           ('specular_map',torch.zeros(1,2,2,4,device=self.device))]:
            with self.subTest(name=name),self.assertRaises(ValueError):self.compose(**{name:value})
        with self.assertRaisesRegex(ValueError,'batches'):
            self.compose(image=self.image.expand(2,-1,-1,-1),lighting_map=self.light.expand(3,-1,-1,-1))

    def test_invalid_scalar_and_unknown_enum_rejected(self):
        for kw in ({'lighting_strength':float('nan')},{'ambient_floor':1.1},{'tone_mapping':'bad'}):
            with self.assertRaises(ValueError):self.compose(**kw)

    def test_inputs_are_not_mutated(self):
        before=self.image.clone();tele=self.tele['direct_diffuse'].clone()
        self.compose(room_bounce_strength=.5)
        self.assertTrue(torch.equal(before,self.image));self.assertTrue(torch.equal(tele,self.tele['direct_diffuse']))

    def test_depth_reconstruction_has_per_image_splits(self):
        d2=self.depth.roll(4,2)*2+3
        p,d,s=reconstruct_metric_surface(torch.cat((self.depth,d2)),24,24)
        for i,source in enumerate((self.depth,d2)):
            solo=reconstruct_metric_surface(source,24,24)
            self.assertTrue(torch.allclose(p[i:i+1],solo[0],atol=1e-6))
            self.assertAlmostEqual(float(s[i]),solo[2],places=6)

    def test_rig_exports_matching_raw_passes_and_ambient_without_depth(self):
        normals=torch.tensor([.5,.5,1.],device=self.device).view(1,1,1,3).expand_as(self.image)
        key=dict(enabled=True,azimuth=0.,elevation=20.,distance=1.8,intensity=1.,kelvin=3200,modifier='Hard Snoot')
        off={**key,'enabled':False}
        light,_,diff,spec=render_studio_lighting(normals,None,key,off,off,ambient_intensity=.03,return_passes=True,_render_probe=False)
        t=light._studiolight_telemetry
        self.assertTrue(torch.equal(t['direct_diffuse'],diff));self.assertTrue(torch.equal(t['direct_specular'],spec))
        self.assertTrue(torch.allclose(light,t['direct_diffuse']+t['direct_specular']+t['ambient_rgb'],atol=1e-6))

    def test_new_comfy_entrypoint_preserves_legacy_python_api(self):
        # Stub only the expensive rig call; this test verifies output/API packaging.
        rig=StudioLightRig();self.light._studiolight_telemetry=self.tele
        legacy=(self.light,self.light,'prompt',self.light)
        rig.execute_rig=lambda **kwargs:legacy
        self.assertEqual(len(rig.execute_rig()),4)
        result=getattr(rig,rig.FUNCTION)()
        self.assertEqual(len(result),len(rig.RETURN_TYPES));self.assertIs(result[4],self.tele)

    def test_workflow_diagnostic_link_is_reachable(self):
        workflow=json.loads((Path(__file__).parent/'workflows/StudioLight_Master.json').read_text())
        nodes={n['id']:n for n in workflow['nodes']}
        diagnostics=[l for l in workflow['links'] if l[1:3]==[17,1]]
        self.assertEqual(len(diagnostics),1)
        self.assertEqual(nodes[diagnostics[0][3]]['type'],'PreviewImage')

if __name__=='__main__':unittest.main()
