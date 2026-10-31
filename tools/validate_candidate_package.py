"""Dependency-free package/graph checks. Does not execute models or certify GPU quality."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def validate(ui, api, frozen):
    nodes={n['id']:n for n in ui['nodes']}
    links={l[0]:l for l in ui['links']}
    assert len(links)==len(ui['links']), 'Duplicate link IDs'
    used=set()
    for l in links.values():
        _,src,slot,dst,inp,typ=l
        assert (dst,inp) not in used, 'Duplicate destination'
        used.add((dst,inp))
        a=nodes[src]['outputs'][slot]; b=nodes[dst]['inputs'][inp]
        assert l[0] in a['links'] and b['link']==l[0], 'Link endpoints disagree'
        assert a['type'] in (typ,'*') and b['type'] in (typ,'*'), 'Link type mismatch'
    def source(node,name):
        i=next(i for i in nodes[node]['inputs'] if i['name']==name)
        assert i['link'] in links, f'Missing input: {node}.{name}'
        l=links[i['link']];return [str(l[1]),l[2]]
    assert source(20,'vae')==['3',0], 'VAE disconnected'
    assert source(20,'pixels')==source(19,'images')==['17',0], 'Preview differs from encoded proxy'
    assert source(17,'diffuse_base')==['92',0], 'Diffuse base contains original background'
    assert source(92,'albedo')==['58',1], 'Wrong albedo output'
    assert nodes[92]['widgets_values']==[1], 'Acceptance requires full albedo strength'
    for k,node in api.items():
        if k.isdigit() and int(k) in nodes:
            un=nodes[int(k)]
            assert un['type']==node['class_type'], f'UI/API class mismatch {k}'
            for name,v in node['inputs'].items():
                if isinstance(v,list) and len(v)==2 and isinstance(v[1],int):
                    assert source(int(k),name)==v, f'UI/API connection mismatch {k}.{name}'
    for graph in (api,frozen):
        active=set();done=set()
        def visit(k):
            assert k not in active, 'Graph cycle'
            if k in done:return
            active.add(k)
            for v in graph[k]['inputs'].values():
                if isinstance(v,list) and len(v)==2 and isinstance(v[1],int):
                    assert str(v[0]) in graph, 'Missing API dependency'
                    visit(str(v[0]))
            active.remove(k);done.add(k)
        for k in graph:visit(k)
        assert graph['17']['inputs']['diffuse_base']==['92',0]
        assert graph['consistent_diffuse_base']['inputs']['images']==['92',0]
        assert graph['58']['inputs']['allow_download'] is False
        assert graph['92']['inputs']['strength']==1
    assert frozen['12']['class_type']=='LoadImage'
    assert not any(n['class_type'] in ('KSampler','KSamplerAdvanced','UNETLoader','DualCLIPLoader','VAEEncode','VAEDecode') for n in frozen.values()), 'Proxy-only graph executes diffusion'
    assert nodes[58]['widgets_values_named']['allow_download'] is False
    for k in (8,15,17):
        for name,value in nodes[k]['widgets_values_named'].items():
            assert api[str(k)]['inputs'][name]==value, f'Stale named widget {k}.{name}'
    return True

def load():
    return tuple(json.loads((ROOT/p).read_text()) for p in (
        'user/default/workflows/StudioLight_Consistent_V1.json',
        'user/default/studiolight/StudioLight_Consistent_V1_API.json',
        'user/default/studiolight/StudioLight_Consistent_V1_FixedSource_Proxy_API.json'))

if __name__=='__main__':
    validate(*load())
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
    m=json.loads((ROOT/'MANIFEST.json').read_text())
    for name,digest in m['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Checksum mismatch: '+name
    print('PASS: UI/API wiring, required VAE, exact preview, full-surface albedo, frozen-source pruning, widget parity, Python syntax and manifest.')
    print('GPU runtime, model availability and visual quality remain pending.')
