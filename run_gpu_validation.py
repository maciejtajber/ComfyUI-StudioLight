#!/usr/bin/env python3
"""
GPU Validation Runner for ComfyUI-StudioLight
Connects to local ComfyUI tunnel (http://127.0.0.1:8188) and executes
benchmark test cases with full progress monitoring and output collection.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import websocket

COMFY_HOST = "127.0.0.1:8188"
WORKFLOW_PATH = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_Master_v2_Clean.json"
OUTPUT_DIR = "/Users/maciejtajber/Desktop/StudioLight_GPU_Validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_object_info():
    req = urllib.request.urlopen(f"http://{COMFY_HOST}/object_info")
    return json.loads(req.read().decode())

def ui_to_api(wf_path, object_info):
    with open(wf_path) as f:
        wf = json.load(f)

    links = {l[0]: (l[1], l[2]) for l in wf.get("links", [])}
    nodes = wf.get("nodes", [])

    prompt = {}
    for node in nodes:
        nid = str(node["id"])
        ntype = node["type"]
        if ntype not in object_info:
            continue

        info = object_info[ntype]
        inputs_spec = {}
        inputs_spec.update(info.get("input", {}).get("required", {}))
        inputs_spec.update(info.get("input", {}).get("optional", {}))

        node_inputs = {}
        linked_slots = {}
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            if link_id is not None and link_id in links:
                from_node, from_slot = links[link_id]
                node_inputs[inp["name"]] = [str(from_node), from_slot]
                linked_slots[inp["name"]] = True

        widgets = node.get("widgets_values", [])
        if isinstance(widgets, list):
            widget_idx = 0
            for name, spec in inputs_spec.items():
                if name not in linked_slots:
                    if widget_idx < len(widgets):
                        node_inputs[name] = widgets[widget_idx]
                        widget_idx += 1
        elif isinstance(widgets, dict):
            for k, v in widgets.items():
                if k not in linked_slots:
                    node_inputs[k] = v

        prompt[nid] = {
            "class_type": ntype,
            "inputs": node_inputs
        }

    return prompt

def queue_prompt(prompt, client_id="gpu_val_runner"):
    p_data = json.dumps({"prompt": prompt, "client_id": client_id}).encode('utf-8')
    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=p_data, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read().decode())

def wait_for_completion(prompt_id, client_id="gpu_val_runner", timeout=300):
    ws_url = f"ws://{COMFY_HOST}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url, timeout=timeout)
    print(f"Connected to ComfyUI websocket. Waiting for prompt_id: {prompt_id} ...")

    start_time = time.time()
    last_node = ""

    while time.time() - start_time < timeout:
        out = ws.recv()
        if isinstance(out, str):
            msg = json.loads(out)
            mtype = msg.get("type")
            data = msg.get("data", {})

            if mtype == "executing":
                node = data.get("node")
                if node is None and data.get("prompt_id") == prompt_id:
                    print(f"\nExecution of {prompt_id} complete in {time.time() - start_time:.1f}s!")
                    ws.close()
                    return True
                elif node != last_node:
                    last_node = node
                    print(f" -> Executing Node {node}...", end="", flush=True)

            elif mtype == "progress":
                val = data.get("value", 0)
                mx = data.get("max", 1)
                print(f"\r -> Executing Node {last_node} [{val}/{mx}]... ", end="", flush=True)

            elif mtype == "execution_error":
                print(f"\nExecution Error on prompt {prompt_id}:")
                print(json.dumps(data, indent=2))
                ws.close()
                return False

    ws.close()
    raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout}s")

def get_history(prompt_id):
    req = urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}")
    return json.loads(req.read().decode()).get(prompt_id, {})

def download_images(history, case_name):
    outputs = history.get("outputs", {})
    saved_files = []
    for nid, out_dict in outputs.items():
        images = out_dict.get("images", [])
        for img_info in images:
            fname = img_info.get("filename")
            subfolder = img_info.get("subfolder", "")
            ftype = img_info.get("type", "output")
            url = f"http://{COMFY_HOST}/view?filename={urllib.parse.quote(fname)}&subfolder={urllib.parse.quote(subfolder)}&type={ftype}"
            local_fname = f"{case_name}_{nid}_{fname}"
            local_path = os.path.join(OUTPUT_DIR, local_fname)
            print(f"Downloading output {fname} (Node {nid}) -> {local_path} ...")
            urllib.request.urlretrieve(url, local_path)
            saved_files.append(local_path)
    return saved_files

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "techno_girl"
    object_info = get_object_info()
    prompt = ui_to_api(WORKFLOW_PATH, object_info)

    print(f"Loaded workflow template. Preparing Case: {case} ...")

    if case == "techno_girl":
        # Case 2: Techno Girl (00053_)
        prompt["6"]["inputs"]["value"] = "Cinematic fashion editorial in a dark underground techno club, striking young woman with wet slicked-back hair, intense lighting, green laser cuts in background, you bearly notcie people dancing in the smoke iluminated band sliced by green laser and cold strobe light in the cackground, deep ambient shadows, 35mm lens"
        prompt["7"]["inputs"]["value"] = ", flat overcast diffuse daylight, soft even shadowless lighting, rich natural color, beauty catalog"
        prompt["11"]["inputs"]["noise_seed"] = 15021561791463
        prompt["11"]["inputs"]["cfg"] = 2.2
        prompt["11"]["inputs"]["steps"] = 30
        prompt["11"]["inputs"]["start_at_step"] = 0
        prompt["11"]["inputs"]["end_at_step"] = 10
        prompt["25"]["inputs"]["seed"] = 424242
        prompt["25"]["inputs"]["steps"] = 30
        prompt["25"]["inputs"]["cfg"] = 1.0
        prompt["25"]["inputs"]["denoise"] = 0.6
        prompt["15"]["inputs"]["preset"] = "Direct On-Camera Flash"
        prompt["15"]["inputs"]["key_azimuth"] = 0.0
        prompt["15"]["inputs"]["key_elevation"] = 5.0
        prompt["15"]["inputs"]["key_intensity"] = 1.4
        prompt["15"]["inputs"]["key_kelvin"] = 5500
        prompt["15"]["inputs"]["key_modifier"] = "Direct On-Camera Flash"
        prompt["15"]["inputs"]["fill_enabled"] = False
        prompt["15"]["inputs"]["rim_enabled"] = False
        prompt["27"]["inputs"]["filename_prefix"] = f"GPUVal_{case}"

    elif case == "lake_girl":
        # Case 5: Lake Girl (00024_)
        prompt["6"]["inputs"]["value"] = "young girl swimming in the lake late at night in nature, a close up"
        prompt["7"]["inputs"]["value"] = "auto"
        prompt["11"]["inputs"]["noise_seed"] = 125
        prompt["11"]["inputs"]["cfg"] = 2.2
        prompt["11"]["inputs"]["steps"] = 30
        prompt["11"]["inputs"]["start_at_step"] = 0
        prompt["11"]["inputs"]["end_at_step"] = 10
        prompt["25"]["inputs"]["seed"] = 424242
        prompt["25"]["inputs"]["steps"] = 30
        prompt["25"]["inputs"]["cfg"] = 1.0
        prompt["25"]["inputs"]["denoise"] = 0.55
        prompt["15"]["inputs"]["preset"] = "Direct On-Camera Flash"
        prompt["15"]["inputs"]["key_azimuth"] = 0.0
        prompt["15"]["inputs"]["key_elevation"] = 5.0
        prompt["15"]["inputs"]["key_intensity"] = 1.4
        prompt["15"]["inputs"]["key_kelvin"] = 5500
        prompt["15"]["inputs"]["key_modifier"] = "Direct On-Camera Flash"
        prompt["15"]["inputs"]["fill_enabled"] = False
        prompt["15"]["inputs"]["rim_enabled"] = False
        prompt["27"]["inputs"]["filename_prefix"] = f"GPUVal_{case}"

    elif case == "pool_woman":
        # Case 6: Pool Woman (00027_)
        prompt["6"]["inputs"]["value"] = "a half naked model posing in the swimming pool at night"
        prompt["7"]["inputs"]["value"] = "auto"
        prompt["11"]["inputs"]["noise_seed"] = 125
        prompt["11"]["inputs"]["cfg"] = 2.2
        prompt["11"]["inputs"]["steps"] = 30
        prompt["11"]["inputs"]["start_at_step"] = 0
        prompt["11"]["inputs"]["end_at_step"] = 10
        prompt["25"]["inputs"]["seed"] = 424242
        prompt["25"]["inputs"]["steps"] = 30
        prompt["25"]["inputs"]["cfg"] = 1.0
        prompt["25"]["inputs"]["denoise"] = 0.55
        prompt["15"]["inputs"]["preset"] = "Golden Hour Window"
        prompt["15"]["inputs"]["key_azimuth"] = -60.0
        prompt["15"]["inputs"]["key_elevation"] = 15.0
        prompt["15"]["inputs"]["key_intensity"] = 1.3
        prompt["15"]["inputs"]["key_kelvin"] = 3000
        prompt["15"]["inputs"]["key_modifier"] = "Octabox (Soft)"
        prompt["15"]["inputs"]["fill_enabled"] = True
        prompt["15"]["inputs"]["fill_azimuth"] = 45.0
        prompt["15"]["inputs"]["fill_elevation"] = 20.0
        prompt["15"]["inputs"]["fill_intensity"] = 0.35
        prompt["15"]["inputs"]["fill_kelvin"] = 6500
        prompt["15"]["inputs"]["fill_modifier"] = "Octabox (Soft)"
        prompt["15"]["inputs"]["rim_enabled"] = True
        prompt["15"]["inputs"]["rim_azimuth"] = -135.0
        prompt["15"]["inputs"]["rim_elevation"] = 25.0
        prompt["15"]["inputs"]["rim_intensity"] = 0.9
        prompt["15"]["inputs"]["rim_kelvin"] = 3200
        prompt["15"]["inputs"]["rim_modifier"] = "Stripbox"
        prompt["27"]["inputs"]["filename_prefix"] = f"GPUVal_{case}"

    elif case == "car_girl":
        # Case 8: Woman in Car (00035_)
        prompt["6"]["inputs"]["value"] = "a girl with the car at dusk"
        prompt["7"]["inputs"]["value"] = "auto"
        prompt["11"]["inputs"]["noise_seed"] = 125
        prompt["11"]["inputs"]["cfg"] = 2.2
        prompt["11"]["inputs"]["steps"] = 30
        prompt["11"]["inputs"]["start_at_step"] = 0
        prompt["11"]["inputs"]["end_at_step"] = 10
        prompt["25"]["inputs"]["seed"] = 424242
        prompt["25"]["inputs"]["steps"] = 30
        prompt["25"]["inputs"]["cfg"] = 1.0
        prompt["25"]["inputs"]["denoise"] = 0.55
        prompt["15"]["inputs"]["preset"] = "Custom"
        prompt["15"]["inputs"]["key_azimuth"] = 40.0
        prompt["15"]["inputs"]["key_elevation"] = 5.0
        prompt["15"]["inputs"]["key_intensity"] = 1.4
        prompt["15"]["inputs"]["key_kelvin"] = 7200
        prompt["15"]["inputs"]["key_modifier"] = "Direct On-Camera Flash"
        prompt["15"]["inputs"]["fill_enabled"] = False
        prompt["15"]["inputs"]["rim_enabled"] = False
        prompt["27"]["inputs"]["filename_prefix"] = f"GPUVal_{case}"

    else:
        print(f"Unknown case: {case}")
        sys.exit(1)

    res = queue_prompt(prompt)
    prompt_id = res.get("prompt_id")
    print(f"Prompt successfully queued! prompt_id: {prompt_id}")

    ok = wait_for_completion(prompt_id)
    if ok:
        history = get_history(prompt_id)
        saved = download_images(history, case)
        print(f"\nAll outputs retrieved successfully for {case}:")
        for s in saved:
            print("  *", s)
