import unittest,os
import torch
from nodes.photometric_composite import StudioPhotometricComposite,soft_light_blend

class SoftLightTests(unittest.TestCase):
 def setUp(self):
  self.device=os.environ.get('STUDIOLIGHT_TEST_DEVICE','cpu')
  self.image=torch.tensor([.05,.2,.45,.7,.9,.99],device=self.device).view(1,1,2,3)
  self.node=StudioPhotometricComposite()
 def run_mode(self,gain,**kwargs):
  return self.node.composite(self.image,torch.full_like(self.image,gain),compositing_mode='Soft Light',lighting_strength=1.,**kwargs)[0]
 def test_neutral_gain_is_identity(self):
  for amount in [0.,.35,1.]:
   self.assertTrue(torch.allclose(self.run_mode(1.,blend_amount=amount),self.image,atol=1e-6,rtol=0))
 def test_zero_blend_is_original_even_with_specular(self):
  self.assertTrue(torch.allclose(self.run_mode(5.,blend_amount=0.,specular_map=torch.full_like(self.image,.3),lighting_map_includes_specular=False,specular_kick=1.),self.image,atol=1e-6,rtol=0))
 def test_shadow_and_highlight_direction_and_hdr_range(self):
  self.assertTrue((self.run_mode(.2)<=self.image+1e-7).all())
  self.assertTrue((self.run_mode(8.)>=self.image-1e-7).all())
  for g in [0.,.2,1.,10000.]:
   a=self.run_mode(g,blend_amount=1.)
   self.assertTrue(torch.isfinite(a).all() and (a>=0).all() and (a<=1).all())
 def test_combined_pass_and_diffuse_only_count_specular_once(self):
  d=torch.full_like(self.image,.4);s=torch.full_like(d,.2)
  a=self.node.composite(self.image,d+s,specular_map=s,compositing_mode='Soft Light',specular_kick=.45)[0]
  b=self.node.composite(self.image,d,specular_map=s,lighting_map_includes_specular=False,compositing_mode='Soft Light',specular_kick=.45)[0]
  self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=0))
 def test_default_linear_mode_remains_explicit_linear_mode(self):
  g=torch.full_like(self.image,.5)
  self.assertTrue(torch.equal(self.node.composite(self.image,g)[0],self.node.composite(self.image,g,compositing_mode='Linear HDR')[0]))
 def test_unknown_mode_refused(self):
  with self.assertRaisesRegex(ValueError,'compositing mode'):self.node.composite(self.image,torch.ones_like(self.image),compositing_mode='unknown')

if __name__=='__main__':unittest.main()
