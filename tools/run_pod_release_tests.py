#!/usr/bin/env python3
"""Automated release gate and fixed-seed 2x2 sanitizer/HDR visual experiment."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from urllib.parse import urlencode

ROOT=Path(__file__).resolve().parent.parent

def request(base,path,data=None):
    req=Request(base.rstrip('/')+path,data=None if data is None else json.dumps(data).encode(),
                headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=60) as response: return json.load(response)

def to_api(wf,info):
    """Convert named serialized widget values and links; inline primitive strings."""
    nodes={n['id']:n for n in wf['nodes']}
    edges={l[0]:l for l in wf['links']}
    ignored={'MarkdownNote','PreviewImage','PreviewAny','PrimitiveStringMultiline'}
    graph={}
    for nid,n in nodes.items():
        kind=n['type']
        if kind in ignored: continue
        if kind not in info: raise ValueError(f'Missing node class: {kind}')
        schema=info[kind]['input']
        allowed={**schema.get('required',{}),**schema.get('optional',{})}
        values={k:v for k,v in n.get('widgets_values_named',{}).items() if k in allowed}
        for inp in n.get('inputs',[]):
            if inp.get('link') is not None:
                _,src,slot,_,_,_=edges[inp['link']]
                source=nodes[src]
                values[inp['name']]=source['widgets_values'][0] if source['type']=='PrimitiveStringMultiline' else [str(src),slot]
        missing=set(schema.get('required',{}))-set(values)
        for name in list(missing):
            spec=allowed[name]
            if len(spec)>1 and 'default' in spec[1]: values[name]=spec[1]['default'];missing.remove(name)
        if missing: raise ValueError(f'{kind} #{nid} missing inputs: {sorted(missing)}')
        for name,spec in allowed.items():
            value=values.get(name)
            if isinstance(spec[0],list) and value is not None and not (isinstance(value,list) and len(value)==2):
                if value not in spec[0]: raise ValueError(f'{kind}.{name}={value!r} not available on pod; select an explicit --unet/--t5 override if necessary')
        graph[str(nid)]={'class_type':kind,'inputs':values}
    return graph

def arms():
    rows=[]
    for seed in (777777,424242):
        for preset in ('Cinematic Film Noir','High-Fashion Beauty','Direct On-Camera Flash'):
            for sanitizer in (False,True):
                for hdr in (False,True):
                    rows.append(dict(seed=seed,preset=preset,sanitizer=sanitizer,hdr=hdr,stress=False))
    for seed in (777777,424242):
        for hdr in (False,True):
            rows.append(dict(seed=seed,preset='Custom',sanitizer=True,hdr=hdr,stress=True))
    for i,row in enumerate(rows): row['id']=f'case_{i+1:02d}'
    return rows

def build_case(template,wf,row,session):
    g=copy.deepcopy(template)
    n={v['id']:v for v in wf['nodes']}
    if not row['sanitizer']:
        g['9']['inputs']['text']=n[6]['widgets_values'][0]+n[7]['widgets_values'][0]
        g['21']['inputs']['string_a']=n[6]['widgets_values'][0]
        g.pop('8',None)
    g['11']['inputs']['noise_seed']=row['seed']
    g['25']['inputs']['seed']=row['seed']
    g['15']['inputs']['preset']=row['preset']
    if row['stress']:
        g['15']['inputs'].update(key_enabled=True,key_azimuth=0.,key_elevation=20.,key_distance=.8,
                                key_intensity=3.,key_kelvin=3200,key_modifier='Beauty Dish (Silver)',
                                fill_enabled=False,rim_enabled=False)
    if not row['hdr']:
        g['80']={'class_type':'StudioLightTestClamp','inputs':{'image':['15',0]}}
        g['81']={'class_type':'StudioLightTestClamp','inputs':{'image':['15',3]}}
        g['17']['inputs']['lighting_map']=['80',0]
        g['17']['inputs']['specular_map']=['81',0]
    prefix=f'StudioLightRelease/{session}/{row["id"]}'
    g['27']['inputs']['filename_prefix']=prefix+'/final'
    captures={'draft':['12',0],'depth':['29',0],'normals':['14',0],
              'lighting_raw':['15',0],'specular_raw':['15',3],
              'lighting_used':g['17']['inputs']['lighting_map'],
              'specular_used':g['17']['inputs']['specular_map'],
              'composite':['17',0],'darkroom':['18',0],'final':['26',0]}
    for i,(name,link) in enumerate(captures.items(),100):
        g[str(i)]={'class_type':'StudioLightTestCapture','inputs':{'image':link,'prefix':prefix+'/'+name}}
    return g

def download(base,file,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(base.rstrip('/')+'/view?'+urlencode(file),timeout=60) as response:
        destination.write_bytes(response.read())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8188')
    p.add_argument('--out',type=Path,default=ROOT/'release_test_results')
    p.add_argument('--unet');p.add_argument('--t5')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--timeout',type=int,default=1800,help='Per-render seconds before stopping; queued job is not cancelled')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    wf=json.loads((ROOT/'workflows/StudioLight_Master.json').read_text())
    matrix=arms()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest={'commit':commit,'workflow_sha256':hashlib.sha256(json.dumps(wf,sort_keys=True).encode()).hexdigest(),
              'unet_override':a.unet,'t5_override':a.t5,'cases':matrix}
    manifest_path=a.out/'manifest.json'
    if manifest_path.exists():
        if not a.resume: raise SystemExit('Output folder already contains a manifest. Use a new --out or --resume.')
        if json.loads(manifest_path.read_text())!=manifest: raise SystemExit('Resume manifest differs from this code/workflow/experiment. Use a new output folder.')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    if a.plan_only:
        print(f'Prepared {len(matrix)} cases in {manifest_path}');return
    print('Running local CUDA numerical gates...',flush=True)
    env={**os.environ,'STUDIOLIGHT_TEST_DEVICE':'cuda'}
    with (a.out/'numerical-tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','test_workflow_migration','test_release_candidate','test_pod_harness','-v'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'test_studiolight.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    info=request(a.url,'/object_info')
    for kind in ('StudioLightTestClamp','StudioLightTestCapture'):
        if kind not in info: raise SystemExit('Test helpers not loaded. Run prepare_pod_release_tests.py before starting/restarting ComfyUI.')
    stats=request(a.url,'/system_stats')
    if not any('cuda' in str(d.get('type','')).lower() for d in stats.get('devices',[])):
        raise SystemExit('ComfyUI server is not reporting a CUDA device')
    (a.out/'server-system-stats.json').write_text(json.dumps(stats,indent=2))
    queue=request(a.url,'/queue')
    if queue.get('queue_running') or queue.get('queue_pending'): raise SystemExit('ComfyUI queue is busy; release tests will not interfere with existing work')
    if a.unet:
        next(n for n in wf['nodes'] if n['id']==1)['widgets_values_named']['unet_name']=a.unet
    if a.t5:
        next(n for n in wf['nodes'] if n['id']==2)['widgets_values_named']['clip_name2']=a.t5
    template=to_api(wf,info)
    session=a.out.resolve().name
    for row in matrix:
        directory=a.out/row['id'];done=directory/'completed.json'
        if a.resume and done.exists(): continue
        directory.mkdir(exist_ok=True)
        graph=build_case(template,wf,row,session)
        (directory/'prompt.json').write_text(json.dumps(graph,indent=2))
        print(f'{row["id"]}/{len(matrix)}: {row}',flush=True)
        accepted=request(a.url,'/prompt',{'prompt':graph,'client_id':'studiolight-release-tests'})
        if accepted.get('node_errors'): raise RuntimeError(accepted['node_errors'])
        pid=accepted['prompt_id'];(directory/'queued.json').write_text(json.dumps({'prompt_id':pid}))
        started=time.monotonic()
        while True:
            history=request(a.url,'/history/'+pid).get(pid)
            if history:
                (directory/'history.json').write_text(json.dumps(history,indent=2))
                if history.get('status',{}).get('status_str')=='error': raise RuntimeError(f'{row["id"]} failed; inspect history.json')
                if history.get('status',{}).get('completed'): break
            if time.monotonic()-started>a.timeout: raise TimeoutError(f'{pid} exceeded timeout; inspect the server before resuming. No job was cancelled.')
            time.sleep(3)
        files=[]
        for output in history.get('outputs',{}).values():
            for file in output.get('studiolight_files',[])+output.get('images',[]):
                destination=directory/file['filename'];download(a.url,file,destination);files.append(destination.name)
        if not (directory/'final.npz').exists(): raise RuntimeError('Final float capture missing')
        done.write_text(json.dumps({'case':row,'prompt_id':pid,'files':files},indent=2))
    import numpy as np
    gates=[]
    for row in matrix:
        directory=a.out/row['id']
        for raw_name,used_name in (('lighting_raw','lighting_used'),('specular_raw','specular_used')):
            raw=np.load(directory/(raw_name+'.npz'))['image']
            used=np.load(directory/(used_name+'.npz'))['image']
            expected=raw if row['hdr'] else np.clip(raw,0,1)
            if not np.allclose(used,expected,atol=1e-6,rtol=0):
                raise AssertionError(f"{row['id']} {used_name} does not match its HDR arm")
        if row['hdr']:
            baseline=next(b for b in matrix if not b['hdr'] and all(b[k]==row[k] for k in ('seed','preset','sanitizer','stress')))
            for name in ('draft','depth','normals'):
                corrected=np.load(directory/(name+'.npz'))['image']
                old=np.load(a.out/baseline['id']/(name+'.npz'))['image']
                delta=float(np.max(np.abs(corrected-old)))
                gates.append({'case':row['id'],'baseline':baseline['id'],'pass':name,'max_abs_difference':delta})
                if delta>1e-6: raise AssertionError(f"HDR pair changed {name}: {delta}")
    (a.out/'comparison_gates.json').write_text(json.dumps(gates,indent=2))
    from PIL import Image,ImageDraw
    panels=[]
    for row in matrix:
        directory=a.out/row['id'];pngs=sorted(directory.glob('final_*.png'))
        if not pngs: raise RuntimeError(f'Missing display final for {row["id"]}')
        im=Image.open(pngs[0]).convert('RGB');im.thumbnail((384,384))
        panel=Image.new('RGB',(384,430),'#181818');panel.paste(im,(0,0))
        ImageDraw.Draw(panel).text((8,388),f'{row["id"]} seed {row["seed"]}\nS={int(row["sanitizer"])} HDR={int(row["hdr"])} {row["preset"]}',fill='white')
        panels.append(panel)
    sheet=Image.new('RGB',(384*4,430*7),'#181818')
    for i,im in enumerate(panels): sheet.paste(im,((i%4)*384,(i//4)*430))
    sheet.save(a.out/'contact_sheet.png')
    summary=[]
    for row in matrix:
        directory=a.out/row['id']
        summary.append({**row,'lighting':json.loads((directory/'lighting_raw.json').read_text()),
                        'composite':json.loads((directory/'composite.json').read_text()),
                        'final':json.loads((directory/'final.json').read_text())})
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('All numerical and execution gates passed. Visual review remains required; inspect contact_sheet.png and summary.json.',flush=True)

if __name__=='__main__': main()
