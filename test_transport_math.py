"""Independent analytical light-transport regressions, without generative renders."""
import os,unittest
import torch
from nodes.shader_engine import compute_light_shading,render_studio_lighting
from nodes.photometric_composite import StudioPhotometricComposite,srgb_to_linear

class TransportMathTests(unittest.TestCase):
 def setUp(self):self.device=torch.device(os.environ.get('STUDIOLIGHT_TEST_DEVICE','cpu'))
 def mirror(self,x):
  p=torch.tensor([[[[x,0.,-2.]]]],device=self.device)
  n=torch.tensor([[[[0.,0.,1.]]]],device=self.device)
  v=-p/torch.linalg.vector_norm(p,dim=-1,keepdim=True)
  r=2*(n*v).sum(-1,keepdim=True)*n-v
  d=torch.full((1,1,1,1),.5,device=self.device)
  return compute_light_shading(n,r.reshape(3),(1.,1.,1.),1.,'Direct On-Camera Flash',depth=d,distance=2.,target_pos=p.reshape(1,3),precomputed_surface=(p,d,.3),return_separate=True)[1]
 def test_off_axis_mirror_reflects_emitter_on_analytical_ray(self):
  self.assertGreater(float(self.mirror(1.).max()),.1)
 def test_mirror_emitter_center_reflection_is_not_screen_position_dependent(self):
  self.assertTrue(torch.allclose(self.mirror(0.),self.mirror(1.),atol=1e-6,rtol=0))
 def test_exported_components_account_for_combined_cavity_attenuation(self):
  x=torch.linspace(0,4*torch.pi,32,device=self.device).view(1,1,32,1).expand(1,32,32,1)
  n=torch.cat((.5+.3*torch.sin(x),torch.ones_like(x)*.5,torch.ones_like(x)),dim=-1)
  k=dict(enabled=True,azimuth=30.,elevation=20.,distance=1.8,intensity=1.,kelvin=5500,modifier='Direct On-Camera Flash');off={**k,'enabled':False}
  light,_,diff,spec=render_studio_lighting(n,None,k,off,off,ambient_intensity=.05,normal_boost=1.,ao_anchoring=.8,return_passes=True,_render_probe=False)
  self.assertTrue(torch.allclose(light,diff+spec+.05,atol=1e-6,rtol=0))
 def test_combined_and_diffuse_only_inputs_produce_same_single_specular_result(self):
  base=torch.full((1,2,2,3),.2,device=self.device);diff=torch.full_like(base,.2);spec=torch.full_like(base,.1)
  c=StudioPhotometricComposite()
  combined=c.composite(base,diff+spec,specular_map=spec,specular_kick=.45,lighting_map_includes_specular=True)[0]
  separate=c.composite(base,diff,specular_map=spec,specular_kick=.45,lighting_map_includes_specular=False)[0]
  self.assertTrue(torch.allclose(combined,separate,atol=1e-7,rtol=0))
 def test_specular_disabled_does_not_leave_specular_in_diffuse_product(self):
  base=torch.full((1,1,1,3),.2,device=self.device);diff=torch.full_like(base,.2);spec=torch.full_like(base,.1)
  actual=srgb_to_linear(StudioPhotometricComposite().composite(base,diff+spec,specular_map=spec,specular_kick=0.)[0])
  expected=srgb_to_linear(base)*diff
  self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=0))
 def test_point_fallback_inverse_square_over_full_supported_near_range(self):
  n=torch.tensor([[[[0.,0.,1.]]]],device=self.device);v=torch.tensor([0.,0.,1.],device=self.device)
  def diffuse(distance):return compute_light_shading(n,v,(1.,1.,1.),1.,'Hard Snoot',distance=distance,return_separate=True)[0]
  close,far=diffuse(.3),diffuse(.9)
  self.assertTrue(torch.allclose(close,far*9.,atol=1e-5,rtol=1e-6))
 def test_grey_probe_is_eighteen_percent_linear_reflectance(self):
  n=torch.tensor([.5,.5,1.],device=self.device).view(1,1,1,3).expand(1,8,8,3)
  k=dict(enabled=False,azimuth=0.,elevation=0.,distance=1.8,intensity=0.,kelvin=5500,modifier='Direct On-Camera Flash')
  _,probe,_,_=render_studio_lighting(n,None,k,k,k,ambient_intensity=1.,normal_boost=1.,ao_anchoring=0.,preview_resolution=128,return_passes=True)
  decoded=srgb_to_linear(probe[0,32,32])
  self.assertTrue(torch.allclose(decoded,torch.full_like(decoded,.18),atol=1e-6,rtol=0))
 def test_shared_area_integrator_matches_lambertian_far_field_and_backface(self):
  from nodes.shader_engine import integrate_area_diffuse
  from nodes.color_utils import MODIFIER_PRESETS
  n=torch.tensor([[[[0.,0.,1.]]]],device=self.device);p=torch.zeros_like(n);aim=torch.zeros((1,3),device=self.device)
  def irradiance(distance,normal=n):
   light=torch.tensor([[0.,0.,distance]],device=self.device)
   return integrate_area_diffuse(normal,p,light,aim,MODIFIER_PRESETS['Direct On-Camera Flash'])
  self.assertAlmostEqual(float(irradiance(1.8).max()),1.,delta=.002)
  self.assertAlmostEqual(float(irradiance(3.6).max()),.25,delta=.0005)
  self.assertEqual(float(irradiance(1.8,-n).max()),0.)

if __name__=='__main__':unittest.main()
