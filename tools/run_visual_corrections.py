"""Replay saved float drafts/geometry with unchanged FLUX refinement and explicit corrections."""
from pathlib import Path
import sys,copy,json,time,hashlib
from urllib.request import urlopen,Request
from urllib.parse import urlencode
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline',type=Path,required=True,help='Original 28-case results: case_XX folders directly or under generations/')
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--url',default='http://127.0.0.1:8188')
parser.add_argument('--source-session',default='STUDIOLIGHT_RELEASE_20261013',help='Existing captures under ComfyUI output/StudioLightRelease/')
args=parser.parse_args()
ROOT=args.baseline.resolve()
OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
BASE=args.url.rstrip('/')
GENERATIONS=ROOT/'generations' if (ROOT/'generations').is_dir() else ROOT

def req(path,data=None):
 with urlopen(Request(BASE+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'}),timeout=60) as f:return json.load(f)
rows=[('halo_17',17,0),('halo_19',19,0),('stress_26_ev1',26,-1),('stress_26_ev2',26,-2),('stress_28_ev1',28,-1),('stress_28_ev2',28,-2),('regression_noir_04',4,0),('regression_beauty_08',8,0)]
q=req('/queue')
if q.get('queue_running') or q.get('queue_pending'):raise SystemExit('Queue busy: no tests submitted')
info=req('/object_info')
assert 'StudioLightTestLoad' in info and 'exposure_ev' in info['StudioPhotometricComposite']['input']['optional']
baseline_manifest=next((p for p in (ROOT/'manifest.json',ROOT/'validation/manifest.json') if p.is_file()),None)
baseline_revision=json.loads(baseline_manifest.read_text()).get('commit','unknown') if baseline_manifest else 'unknown'
manifest={'baseline_commit':baseline_revision, 'baseline_directory':str(ROOT),'source_sha256':hashlib.sha256((Path(__file__).resolve().parent.parent/'nodes/photometric_composite.py').read_bytes()).hexdigest(),'cases':rows,'note':'Saved float32 drafts and geometry; original seeds, prompts and refinement settings retained. No early-stage regeneration.'}
output_session=OUT.name+'_'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:12]
manifest['output_session']=output_session
manifest_path=OUT/'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
 raise SystemExit('Correction manifest differs: use a new output folder for changed code or settings')
manifest_path.write_text(json.dumps(manifest,indent=2))
for name,case,ev in rows:
 folder=OUT/name;folder.mkdir(exist_ok=True)
 if (folder/'completed.json').exists():continue
 old=GENERATIONS/f'case_{case:02d}'
 g=json.loads((old/'prompt.json').read_text())
 for nid in list(g):
  if int(nid)>=100:g.pop(nid)
 for nid,stage in [('12','draft'),('14','normals'),('29','depth')]:
  g[nid]={'class_type':'StudioLightTestLoad','inputs':{'file':f'StudioLightRelease/{args.source_session}/case_{case:02d}/{stage}.npz'}}
 # Force unclipped light/specular arrays for every corrected arm.
 g['17']['inputs'].update(lighting_map=['15',0],specular_map=['15',3],exposure_ev=ev,tone_mapping='Luminance HDR',background_protection=1.,depth_map=['29',0])
 prefix='StudioLightCorrections/'+output_session+'/'+name
 g['27']['inputs']['filename_prefix']=prefix+'/final'
 stages={'draft':['12',0],'depth':['29',0],'normals':['14',0],'lighting_raw':['15',0],'specular_raw':['15',3],'lighting_used':['15',0],'specular_used':['15',3],'composite':['17',0],'darkroom':['18',0],'final':['26',0]}
 for i,(stage,link) in enumerate(stages.items(),100):g[str(i)]={'class_type':'StudioLightTestCapture','inputs':{'image':link,'prefix':prefix+'/'+stage}}
 (folder/'prompt.json').write_text(json.dumps(g,indent=2))
 accepted=req('/prompt',{'prompt':g,'client_id':'StudioLight-visual-corrections'})
 if accepted.get('node_errors'):raise RuntimeError(accepted)
 pid=accepted['prompt_id'];print('Running',name,pid,flush=True)
 (folder/'queued.json').write_text(json.dumps({'prompt_id':pid}))
 start=time.monotonic()
 while True:
  h=req('/history/'+pid).get(pid)
  if h and h.get('status',{}).get('status_str')=='error':
   (folder/'history.json').write_text(json.dumps(h,indent=2));raise RuntimeError(name+' execution error')
  if h and h.get('status',{}).get('completed'):break
  if time.monotonic()-start>1200:raise TimeoutError(name)
  time.sleep(3)
 (folder/'history.json').write_text(json.dumps(h,indent=2))
 files=[]
 for result in h['outputs'].values():
  for meta in result.get('studiolight_files',[])+result.get('images',[]):
   p=folder/Path(meta['filename']).name
   with urlopen(BASE+'/view?'+urlencode(meta),timeout=90) as f:p.write_bytes(f.read())
   files.append(p.name)
 import numpy as np
 for stage in ('draft','depth','normals','lighting_raw','specular_raw'):
  before=np.load(old/(stage+'.npz'))['image'];after=np.load(folder/(stage+'.npz'))['image']
  delta=float(np.max(np.abs(before-after)))
  if delta>1e-6:raise AssertionError(f'{name}: {stage} changed by {delta}')
 (folder/'completed.json').write_text(json.dumps({'prompt_id':pid,'baseline_case':case,'exposure_ev':ev,'files':files,'unchanged_draft_geometry_and_raw_lighting':True},indent=2))
 print('Saved',name,flush=True)
(OUT/'runner_exit_code.txt').write_text('0\n')
print('All eight correction renders and exact replay gates passed.',flush=True)
