#!/usr/bin/env python3
import os
import sys
import json
import time
import urllib.request
import subprocess

TARGET_ROOT = "/Users/maciejtajber/Desktop/STUDIOLIGHT_SESSION_20261015"
GENS_DIR = os.path.join(TARGET_ROOT, "generations")
FINALS_DIR = os.path.join(TARGET_ROOT, "final_renders")
TEMP_DIR = os.path.join(TARGET_ROOT, "temp_passes")
WORKFLOWS_DIR = os.path.join(TARGET_ROOT, "workflows")

os.makedirs(GENS_DIR, exist_ok=True)
os.makedirs(FINALS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(WORKFLOWS_DIR, exist_ok=True)

print(f"============================================================")
print(f">>> DOWNLOADING COMPLETE STUDIOLIGHT POD SESSION")
print(f">>> Destination: {TARGET_ROOT}")
print(f"============================================================")

REQUIRED_PASSES = {
    'VFX Probes & Studio Radar': '01_vfx_probes_radar.png',
    'Preview: Pure Relighting Field (VFX Isolation)': '02_pure_relighting_field.png',
    'Preview: 3D Lit Proxy Plate (Prior to FLUX)': '03_lit_proxy_plate.png',
    'Preview: DSINE 3D Normals': '04_dsine_normals.png',
    'Preview: Depth Map': '05_depth_map.png',
    'Preview: Pass 1 Neutral Draft (Step 10)': '06_pass1_neutral_draft.png',
    'FINAL LIT RENDER': '07_final_lit_render.png'
}

# 1. Fetch ComfyUI History
print("\n>>> [1/4] Fetching history from http://127.0.0.1:8188/history...")
req = urllib.request.Request("http://127.0.0.1:8188/history", headers={'User-Agent': 'StudioLight-Downloader/1.0'})
with urllib.request.urlopen(req, timeout=10) as resp:
    history = json.loads(resp.read().decode('utf-8'))

print(f">>> Found {len(history)} prompt execution records in history.")

# 2. Package into generation_XX folders
gen_idx = 1
# Sort prompts by timestamp if available
sorted_prompts = sorted(history.items(), key=lambda x: x[1].get('prompt', [0])[0] if isinstance(x[1].get('prompt'), list) and len(x[1].get('prompt')) > 0 else 0)

for prompt_id, run in sorted_prompts:
    outputs = run.get('outputs', {})
    prompt_data = run.get('prompt', [None, {}, {}])
    prompt = prompt_data[2] if len(prompt_data) > 2 and isinstance(prompt_data[2], dict) else {}
    
    matched_nodes = {}
    for node_id, out in outputs.items():
        if 'images' in out and len(out['images']) > 0:
            node_info = prompt.get(str(node_id), {})
            title = node_info.get('_meta', {}).get('title', '')
            for k, fname in REQUIRED_PASSES.items():
                if k in title:
                    matched_nodes[fname] = out['images'][0]
    
    if len(matched_nodes) == 7:
        folder_name = f"generation_{gen_idx:02d}"
        gen_dir = os.path.join(GENS_DIR, folder_name)
        os.makedirs(gen_dir, exist_ok=True)
        print(f"  Packaging {folder_name} (Prompt {prompt_id[:8]})...")
        
        for fname in sorted(matched_nodes.keys()):
            img_meta = matched_nodes[fname]
            url = f"http://127.0.0.1:8188/view?filename={img_meta['filename']}&subfolder={img_meta['subfolder']}&type={img_meta['type']}"
            target = os.path.join(gen_dir, fname)
            urllib.request.urlretrieve(url, target)
        
        rig_inputs = {}
        master_prompt_text = ""
        for nid, n in prompt.items():
            if n.get('class_type') == 'StudioLightRig':
                rig_inputs = n.get('inputs', {})
            if n.get('class_type') == 'PrimitiveStringMultiline' or 'Master Prompt' in n.get('_meta', {}).get('title', ''):
                master_prompt_text = n.get('inputs', {}).get('string', '')
        
        with open(os.path.join(gen_dir, 'lighting_parameters.json'), 'w') as f:
            json.dump(rig_inputs, f, indent=2)
        
        with open(os.path.join(gen_dir, 'run_metadata.json'), 'w') as f:
            json.dump({'prompt_id': prompt_id, 'prompt_text': master_prompt_text, 'images': matched_nodes, 'lighting': rig_inputs}, f, indent=2)
        
        gen_idx += 1

print(f"\n>>> Packaged {gen_idx - 1} complete 7-pass generations into {GENS_DIR}.")

# Also sync into Desktop/STUDIOLIGHT_GITHUB_PRODUCTION_final/generations/ if user wants
gh_dir = "/Users/maciejtajber/Desktop/STUDIOLIGHT_GITHUB_PRODUCTION_final/generations"
if os.path.exists(gh_dir):
    subprocess.run(["rsync", "-av", GENS_DIR + "/", gh_dir + "/"], check=False)

# 3. RSYNC all raw outputs and temp passes directly from pod
print("\n>>> [2/4] Downloading ALL raw final plates from /app/ComfyUI/output/...")
subprocess.run(["rsync", "-avz", "--progress", "-e", "ssh -p 21611", "root@159.48.242.34:/app/ComfyUI/output/", FINALS_DIR + "/"], check=True)

print("\n>>> [3/4] Downloading ALL intermediate temp passes from /app/ComfyUI/temp/...")
subprocess.run(["rsync", "-avz", "--progress", "-e", "ssh -p 21611", "root@159.48.242.34:/app/ComfyUI/temp/", TEMP_DIR + "/"], check=True)

print("\n>>> [4/4] Syncing user workflows from pod...")
subprocess.run(["rsync", "-avz", "-e", "ssh -p 21611", "root@159.48.242.34:/app/ComfyUI/user/default/workflows/", WORKFLOWS_DIR + "/"], check=True)

# 4. Trigger on-pod Cloudflare R2 backup (save_session)
print("\n>>> Triggering Cloudflare R2 session output backup on pod (Standing Rule 6)...")
subprocess.run(["ssh", "-p", "21611", "root@159.48.242.34", "save_session"], check=False)

print("\n" + "=" * 60)
print(f"🎉 SESSION BACKUP COMPLETE!")
print(f"  • Final Renders:   {len(os.listdir(FINALS_DIR))} files -> {FINALS_DIR}")
print(f"  • Temp Passes:     {len(os.listdir(TEMP_DIR))} files -> {TEMP_DIR}")
print(f"  • 7-Pass Bundles:  {len(os.listdir(GENS_DIR))} generations -> {GENS_DIR}")
print(f"  • Workflows:       {len(os.listdir(WORKFLOWS_DIR))} files -> {WORKFLOWS_DIR}")
print("============================================================")
