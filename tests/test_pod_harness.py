"""Offline checks of all planned API graphs and float capture helpers."""
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
import torch
from tools.run_pod_release_tests import arms,build_case,to_api,candidate_commit

ROOT=Path(__file__).parent.parent
class HarnessTests(unittest.TestCase):
    def test_all_28_cases_have_valid_edges_and_independent_controls(self):
        wf=json.loads((ROOT/'workflows/StudioLight_Master.json').read_text())
        # Offline fixture checks graph conversion, not the real server schema.
        info={}
        for n in wf['nodes']:
            names=set(n.get('widgets_values_named',{}))|{i['name'] for i in n.get('inputs',[])}
            info[n['type']]={'input':{'required':{name:('STRING',) for name in names if name!='control_after_generate'}}}
        template=to_api(wf,info)
        rows=arms();self.assertEqual(len(rows),28)
        self.assertEqual(len({row['id'] for row in rows}),28)
        for row in rows:
            g=build_case(template,wf,row,'offline_test')
            for node in g.values():
                for value in node['inputs'].values():
                    if isinstance(value,list) and len(value)==2:
                        self.assertIn(value[0],g)
            self.assertEqual(g['11']['inputs']['noise_seed'],row['seed'])
            self.assertEqual(g['25']['inputs']['seed'],row['seed'])
            self.assertEqual(g['25']['inputs']['denoise'],.55)
            self.assertEqual('8' in g,row['sanitizer'])
            self.assertEqual(g['17']['inputs']['lighting_map'],['15',0] if row['hdr'] else ['80',0])
            self.assertEqual(g['17']['inputs']['specular_map'],['15',3] if row['hdr'] else ['81',0])
            self.assertEqual(g['17']['inputs']['telemetry'], ['15',4] if row['hdr'] else ['82',0])
            self.assertEqual(sum(n['class_type']=='StudioLightTestCapture' for n in g.values()),10)

    def test_zip_provenance_verifies_files_and_refuses_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);file=root/'code.py';file.write_text('candidate code')
            meta={'commit':'a'*40,'files':{'code.py':hashlib.sha256(file.read_bytes()).hexdigest()}}
            (root/'STUDIOLIGHT_CANDIDATE.json').write_text(json.dumps(meta))
            self.assertEqual(candidate_commit(root),'a'*40)
            file.write_text('different code')
            with self.assertRaises(ValueError): candidate_commit(root)

    def test_capture_retains_hdr_and_rejects_path_escape(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            old=sys.modules.get('folder_paths')
            sys.modules['folder_paths']=types.SimpleNamespace(get_output_directory=lambda:tmp)
            try:
                spec=importlib.util.spec_from_file_location('release_test_nodes',ROOT/'tools/release_test_nodes.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                a=torch.tensor([[[[2.,3.,4.]]]])
                module.StudioLightTestCapture().capture(a,'test/lighting')
                clamped=module.StudioLightTestClampTelemetry().clamp({'direct_diffuse':a,'d_norm':a})[0]
                self.assertEqual(float(clamped['direct_diffuse'].max()),1.)
                self.assertTrue(torch.equal(clamped['d_norm'],a))
                saved=np.load(Path(tmp)/'test/lighting.npz')['image']
                self.assertEqual(float(saved.max()),4.)
                replay=module.StudioLightTestLoad().load('test/lighting.npz')[0]
                self.assertTrue(torch.equal(replay,a))
                with self.assertRaises(ValueError): module.StudioLightTestLoad().load('../escape.npz')
                self.assertEqual(json.loads((Path(tmp)/'test/lighting.json').read_text())['channels_over_one_fraction'],1.)
                with self.assertRaises(ValueError): module.StudioLightTestCapture().capture(a,'../escape')
            finally:
                if old is None: sys.modules.pop('folder_paths',None)
                else: sys.modules['folder_paths']=old

if __name__=='__main__': unittest.main()
