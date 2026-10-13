import os,unittest
import torch
from nodes.photometric_composite import StudioPhotometricComposite,luminance_hdr_tonemap,rgb_to_luminance,distant_background_weight

class VisualCorrectionTests(unittest.TestCase):
 def setUp(self):
  device=os.environ.get('STUDIOLIGHT_TEST_DEVICE','cpu')
  self.device=torch.device(device)
  self.comp=StudioPhotometricComposite()
 def test_monotonic_neutral_highlights_and_finite_extreme_energy(self):
  levels=torch.tensor([0.,.2,.6,.9,1.,2.,4.,16.,10000.],device=self.device)
  x=levels.view(1,1,-1,1).expand(-1,-1,-1,3)
  y=luminance_hdr_tonemap(x)
  self.assertTrue(torch.isfinite(y).all() and (y>=0).all() and (y<=1).all())
  self.assertTrue((y[0,0,1:,0]>y[0,0,:-1,0]).all())
  self.assertTrue(torch.allclose(y[:,:,:3],x[:,:,:3],atol=1e-6))
 def test_hdr_color_ratios_survive_when_inside_display_gamut(self):
  x=torch.tensor([[[[1.6,1.5,1.4]]]],device=self.device)
  y=luminance_hdr_tonemap(x)
  self.assertLess(float(y.max()),1.)
  self.assertTrue(torch.allclose(y[...,0]/y[...,1],x[...,0]/x[...,1],atol=1e-6))
 def test_gamut_compression_preserves_mapped_luminance(self):
  x=torch.tensor([[[[16.,2.,.2]]]],device=self.device)
  y=luminance_hdr_tonemap(x)
  neutral=luminance_hdr_tonemap(rgb_to_luminance(x).expand_as(x))
  self.assertTrue((y>=0).all() and (y<=1).all())
  self.assertTrue(torch.allclose(rgb_to_luminance(y),neutral[...,:1],atol=2e-6))
 def test_exposure_is_linear_and_does_not_modify_input(self):
  img=torch.full((1,2,2,3),.3,device=self.device);light=torch.full_like(img,4.)
  result=self.comp.composite(img,light,exposure_ev=-2.,tone_mapping='Luminance HDR')[0]
  expected=self.comp.composite(img,light/4.,tone_mapping='Luminance HDR')[0]
  self.assertTrue(torch.allclose(result,expected,atol=1e-6))
  self.assertEqual(float(light.max()),4.)
 def test_background_protection_preserves_far_pixels_only(self):
  img=torch.full((1,1,4,3),.7,device=self.device);light=torch.full_like(img,.1)
  depth=torch.tensor([0.,.05,.15,1.],device=self.device).view(1,1,4,1)
  result=self.comp.composite(img,light,depth_map=depth,background_protection=1.)[0]
  unprotected=self.comp.composite(img,light)[0]
  self.assertTrue(torch.equal(result[:,:,:2],img[:,:,:2]))
  self.assertTrue(torch.allclose(result[:,:,2:],unprotected[:,:,2:]))
 def test_constant_depth_does_not_exclude_flat_wall(self):
  d=torch.full((1,4,4,1),.5,device=self.device)
  self.assertEqual(float(distant_background_weight(d,(4,4)).max()),0.)
 def test_depth_pairing_and_required_input_errors(self):
  img=torch.ones((2,2,2,3),device=self.device)
  with self.assertRaisesRegex(ValueError,'batches'):
   self.comp.composite(img,img,depth_map=torch.ones((3,2,2,1),device=self.device))
  with self.assertRaisesRegex(ValueError,'requires a depth_map'):
   self.comp.composite(img,img,background_protection=1.)
 def test_master_pipeline_preserves_hdr_display_headroom(self):
  import json
  from pathlib import Path
  from nodes.si_shadow_highlight import SIShadowHighlight
  wf=json.loads((Path(__file__).parent/'workflows/StudioLight_Master.json').read_text())
  settings=next(n for n in wf['nodes'] if n['id']==18)['widgets_values_named']
  img=torch.full((1,3,3,3),.7,device=self.device)
  light=torch.tensor([2.,3.,4.],device=self.device).view(1,1,3,1).expand_as(img)
  mapped=self.comp.composite(img,light,tone_mapping='Luminance HDR')[0]
  prepared=SIShadowHighlight().adjust(mapped,**settings)[0]
  self.assertTrue(torch.equal(prepared,mapped))
  self.assertTrue((prepared<1).all())
  self.assertTrue((prepared[0,0,1:,0]>prepared[0,0,:-1,0]).all())
  legacy=SIShadowHighlight().adjust(mapped,**{**settings,'highlight_amount':.15})[0]
  self.assertGreater(int((legacy>=1).sum()),0)

 def test_depth_normalization_is_independent_per_item(self):
  a=torch.tensor([0.,.05,.15,1.],device=self.device).view(1,1,4,1)
  b=a*4.+7.
  together=distant_background_weight(torch.cat((a,b)),(1,4))
  self.assertTrue(torch.allclose(together[0],together[1],atol=1e-5))

if __name__=='__main__':unittest.main()
