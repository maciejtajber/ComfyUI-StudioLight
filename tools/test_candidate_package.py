import copy
import unittest
from validate_gpu_candidate import load,validate

class GraphRegressionTests(unittest.TestCase):
    def setUp(self):self.ui,self.api,self.frozen=load()
    def test_candidate(self):self.assertTrue(validate(self.ui,self.api,self.frozen))
    def rewire(self,dst,name,src,slot):
        nodes={n['id']:n for n in self.ui['nodes']}
        inp=next(i for i in nodes[dst]['inputs'] if i['name']==name)
        l=next(l for l in self.ui['links'] if l[0]==inp['link'])
        nodes[l[1]]['outputs'][l[2]]['links'].remove(l[0])
        l[1:3]=[src,slot];nodes[src]['outputs'][slot]['links'].append(l[0])
    def test_rejects_missing_vae(self):
        n=next(n for n in self.ui['nodes'] if n['id']==20)
        i=next(i for i in n['inputs'] if i['name']=='vae');lid=i['link'];i['link']=None
        self.ui['links']=[l for l in self.ui['links'] if l[0]!=lid]
        with self.assertRaisesRegex(AssertionError,'Missing input'):validate(self.ui,self.api,self.frozen)
    def test_rejects_graded_preview(self):
        self.rewire(19,'images',18,0)
        with self.assertRaisesRegex(AssertionError,'Preview differs'):validate(self.ui,self.api,self.frozen)
    def test_rejects_source_background_as_albedo(self):
        self.rewire(17,'diffuse_base',87,0)
        with self.assertRaisesRegex(AssertionError,'original background'):validate(self.ui,self.api,self.frozen)
    def test_rejects_api_drift(self):
        self.api['20']['inputs']['vae']=['1',0]
        with self.assertRaisesRegex(AssertionError,'connection mismatch'):validate(self.ui,self.api,self.frozen)
    def test_rejects_stale_download_flag(self):
        next(n for n in self.ui['nodes'] if n['id']==58)['widgets_values_named']['allow_download']=True
        with self.assertRaises(AssertionError):validate(self.ui,self.api,self.frozen)

if __name__=='__main__':unittest.main()
