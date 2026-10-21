"""CPU/CUDA release gates for HDR, batch pairing and actual master graph wiring."""
import json
import os
from pathlib import Path
import unittest
import torch
from nodes.shader_engine import render_studio_lighting
from nodes.photometric_composite import StudioPhotometricComposite
from nodes.prompt_sanitizer import StudioPromptSanitizer

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        requested = os.environ.get('STUDIOLIGHT_TEST_DEVICE', 'cpu')
        if requested == 'cuda' and not torch.cuda.is_available():
            self.fail('CUDA required but unavailable')
        self.device = torch.device(requested)
        self.n = torch.tensor([.5,.5,1.], device=self.device).view(1,1,1,3).expand(1,24,24,3)
        self.d = torch.full((1,24,24,1), .5, device=self.device)
        self.key = dict(enabled=True,azimuth=0.,elevation=0.,distance=1.8,intensity=3.,kelvin=5500,modifier='Hard Snoot')
        self.off = dict(self.key,enabled=False)

    def render(self, n=None, d=None):
        return render_studio_lighting(self.n if n is None else n, d, self.key,self.off,self.off,
                                      ao_anchoring=0.,return_passes=True,_render_probe=False)

    def test_hdr_energy_and_color_survive_rig_output(self):
        for depth in (None,self.d):
            light,_,diff,spec=self.render(d=depth)
            self.assertTrue(torch.isfinite(light).all())
            self.assertGreater(float(light.max()),1.)
            self.assertTrue(torch.allclose(light,diff+spec+.05,atol=1e-6))
            self.assertGreater(float(spec.max()),1. if depth is None else 0.)

    def test_compositor_recovers_distinct_bright_levels(self):
        base=torch.full((1,1,2,3),.4,device=self.device)
        light=torch.tensor([2.,4.],device=self.device).view(1,1,2,1).expand_as(base)
        comp=StudioPhotometricComposite()
        hdr=comp.composite(base,light,specular_kick=0.)[0]
        old=comp.composite(base,light.clamp(0,1),specular_kick=0.)[0]
        self.assertGreater(float(hdr[0,0,1,0]),float(hdr[0,0,0,0]))
        self.assertEqual(float(old[0,0,1,0]),float(old[0,0,0,0]))
        self.assertTrue(torch.isfinite(hdr).all() and (hdr>=0).all() and (hdr<=1).all())

    def test_mismatched_and_empty_geometry_batches_rejected(self):
        for bn,bd in ((2,3),(3,2),(0,1),(1,0)):
            with self.subTest(bn=bn,bd=bd), self.assertRaisesRegex(ValueError,'batch'):
                self.render(self.n.expand(bn,24,24,3),self.d.expand(bd,24,24,1))

    def test_singleton_geometry_broadcast_matches_solo(self):
        solo=self.render(d=self.d)[0]
        for bn,bd in ((2,1),(1,2),(2,2)):
            result=self.render(self.n.expand(bn,24,24,3),self.d.expand(bd,24,24,1))[0]
            self.assertEqual(result.shape[0],2)
            self.assertTrue(torch.allclose(result,solo.expand_as(result),atol=1e-5))

    def test_compositor_batch_contract(self):
        comp=StudioPhotometricComposite()
        base=torch.full((2,4,4,3),.4,device=self.device)
        wrong=torch.ones(3,4,4,3,device=self.device)
        with self.assertRaisesRegex(ValueError,'batch'): comp.composite(base,wrong)
        with self.assertRaisesRegex(ValueError,'batch'): comp.composite(base,torch.ones_like(base),specular_map=wrong)
        with self.assertRaisesRegex(ValueError,'batch'): comp.composite(base[:1],torch.ones_like(base),specular_map=wrong)
        self.assertEqual(comp.composite(base,torch.ones(1,4,4,3,device=self.device))[0].shape,base.shape)

    def test_master_compositor_defaults_and_optional_connections(self):
        wf=json.loads((Path(__file__).parent/'workflows/StudioLight_Master.json').read_text())
        nodes={n['id']:n for n in wf['nodes']}; edges={l[0]:l for l in wf['links']}
        compositor=nodes[17]
        schema = StudioPhotometricComposite.INPUT_TYPES()
        widgets = {k: v for group in schema.values() for k, v in group.items()
                   if isinstance(v[0], list) or v[0] in ('FLOAT', 'INT', 'BOOLEAN')}
        self.assertEqual(len(compositor['widgets_values']), len(widgets))
        self.assertEqual(dict(zip(widgets, compositor['widgets_values'])), compositor['widgets_values_named'])
        self.assertEqual(compositor['widgets_values_named']['filmic_white_point'], 3.5)
        self.assertEqual(compositor['widgets_values_named']['wall_shadow_density'], 0.65)
        self.assertEqual(compositor['widgets_values_named']['room_bounce_strength'], 0.0)
        self.assertEqual(nodes[11]['widgets_values_named']['end_at_step'],10)
        self.assertEqual(nodes[25]['widgets_values_named']['denoise'],0.55)
        for name,source,slot in [('depth_map',29,0),('specular_map',15,3),('telemetry',15,4)]:
            index=next(i for i,x in enumerate(compositor['inputs']) if x['name']==name)
            edge=edges[compositor['inputs'][index]['link']]
            self.assertEqual(edge[1:5],[source,slot,17,index])

    def test_master_graph_routes_both_sanitizer_outputs(self):
        wf=json.loads((Path(__file__).parent/'workflows/StudioLight_Master.json').read_text())
        nodes={n['id']:n for n in wf['nodes']}
        sanitizer=nodes[8]
        self.assertEqual(sanitizer['type'],'StudioPromptSanitizer')
        edges={l[0]:l for l in wf['links']}
        self.assertEqual(edges[6][1:5],[8,0,9,1])
        self.assertEqual(edges[29][1:5],[8,1,21,0])
        for lid,src,ss,dst,ds,kind in wf['links']:
            self.assertEqual(nodes[dst]['inputs'][ds]['link'],lid)
            self.assertIn(lid,nodes[src]['outputs'][ss]['links'])
            self.assertEqual(nodes[src]['outputs'][ss]['type'],kind)
            self.assertIn(nodes[dst]['inputs'][ds]['type'],(kind,'*'))
        master=nodes[6]['widgets_values'][0]
        draft,final=StudioPromptSanitizer().sanitize(master,nodes[7]['widgets_values'][0])
        # Bug 5 fix: Material nouns (glossy, patent leather, liquid chrome) must pass through untouched
        self.assertIn('patent', draft.lower())
        self.assertIn('patent', final.lower())
        self.assertIn('preserving subject identity', final.lower())
        # Verify VERBATIM_RAW backward compatibility
        _, final_raw = StudioPromptSanitizer().sanitize(master, lighting_policy="VERBATIM_RAW")
        self.assertEqual(final_raw, master.strip())

if __name__=='__main__': unittest.main()
