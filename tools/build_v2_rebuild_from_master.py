#!/usr/bin/env python3
"""
ComfyUI-StudioLight — V2 Rebuild Workflow Generator (From Master Base)
Author: Maciej Tajber // METAMATERIAL

Clones the complete, production-proven StudioLight_Master.json (preserving StudioLightRig,
Studio Radar, SIShadowHighlight, prompts, notes, and samplers), and integrates the
approved 5-Stage Rebuild Architecture:
1. Stage 1: Pass 1 Readable Draft (Steps 0->10)
2. Stage 2: StudioTransportPreflight (Pre-Flight Gate checking face ROI)
3. Stage 3: StudioAppearanceRepair (Masked Appearance Repair before normals)
4. Stage 4: StudioLightRig (3-Point Photographic Console) + StudioTransportWall (Cavity Wall Falloff)
5. Stage 5: Pass 2 Lit Synthesis (KSampler denoise 0.55)
"""

import json
import os
import copy

MASTER_PATH = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_Master.json"
OUTPUT_PATH = "/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_v2_Rebuild.json"

def build_v2():
    with open(MASTER_PATH, "r") as f:
        master = json.load(f)

    v2 = copy.deepcopy(master)
    v2["id"] = "StudioLight_v2_Rebuild"

    # Find highest existing node ID and link ID
    max_node_id = max(n["id"] for n in v2["nodes"])
    max_link_id = max(l[0] for l in v2["links"])
    
    current_node_id = 50
    current_link_id = max_link_id + 1

    # 1. CREATE STAGE 2: PRE-FLIGHT GATE REVIEW
    # Positioned above DSINE in Zone 2
    node_preflight = {
        "id": 50,
        "type": "StudioTransportPreflight",
        "pos": [-400, -250],
        "size": [360, 220],
        "flags": {},
        "order": 15,
        "mode": 0,
        "inputs": [
            {"name": "draft", "type": "IMAGE", "link": None},
            {"name": "normals", "type": "IMAGE", "link": None}
        ],
        "outputs": [
            {"name": "review_overlay", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "review_report", "type": "STRING", "links": [], "slot_index": 1}
        ],
        "properties": {"Node name for S&R": "StudioTransportPreflight"},
        "widgets_values": ["encoded_x_right_y_up_z_toward_camera", 0, 0, 0, 0],
        "title": "Stage 2: Pre-Flight Gate Review",
        "color": "#332211",
        "bgcolor": "#221100"
    }

    node_preflight_prev = {
        "id": 51,
        "type": "PreviewImage",
        "pos": [-400, 10],
        "size": [360, 320],
        "flags": {},
        "order": 16,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": [],
        "title": "Preview: Stage 2 Preflight Diagnostic"
    }

    # 2. CREATE STAGE 3: MASKED APPEARANCE REPAIR
    # Positioned between Node 12 (Draft Decode) and Node 14/29 (DSINE & Depth)
    node_repair = {
        "id": 52,
        "type": "StudioAppearanceRepair",
        "pos": [-400, 370],
        "size": [360, 200],
        "flags": {},
        "order": 17,
        "mode": 0,
        "inputs": [
            {"name": "draft", "type": "IMAGE", "link": None},
            {"name": "review_report", "type": "STRING", "link": None}
        ],
        "outputs": [
            {"name": "repaired_draft", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "repair_mask", "type": "MASK", "links": [], "slot_index": 1},
            {"name": "repair_status", "type": "STRING", "links": [], "slot_index": 2}
        ],
        "properties": {"Node name for S&R": "StudioAppearanceRepair"},
        "widgets_values": ["auto_adaptive_lift", 16, 0.45],
        "title": "Stage 3: Masked Appearance Repair",
        "color": "#183328",
        "bgcolor": "#0d2218"
    }

    node_repair_prev = {
        "id": 53,
        "type": "PreviewImage",
        "pos": [-400, 610],
        "size": [360, 320],
        "flags": {},
        "order": 18,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": [],
        "title": "Preview: Stage 3 Accepted Draft"
    }

    # 3. CREATE STAGE 4: ASTRA CAVITY WALL TRANSPORT
    # Positioned next to StudioLightRig in Zone 3
    node_geometry = {
        "id": 54,
        "type": "StudioTransportGeometry",
        "pos": [450, 710],
        "size": [360, 200],
        "flags": {},
        "order": 28,
        "mode": 0,
        "inputs": [
            {"name": "depth", "type": "IMAGE", "link": None},
            {"name": "subject_mask", "type": "MASK", "link": None}
        ],
        "outputs": [
            {"name": "geometry", "type": "SL_GEOMETRY", "links": [], "slot_index": 0},
            {"name": "assumed_depth_preview", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "report", "type": "STRING", "links": [], "slot_index": 2}
        ],
        "properties": {"Node name for S&R": "StudioTransportGeometry"},
        "widgets_values": ["Inside", "In-Between", "Meters (Depth)", 6.0, 2.0, 50.0, 0.0, 0.0],
        "title": "StudioLight · Assumed Geometry",
        "color": "#243141",
        "bgcolor": "#101d2d"
    }

    node_wall = {
        "id": 55,
        "type": "StudioTransportWall",
        "pos": [450, 950],
        "size": [360, 260],
        "flags": {},
        "order": 29,
        "mode": 0,
        "inputs": [
            {"name": "geometry", "type": "SL_GEOMETRY", "link": None},
            {"name": "normals", "type": "IMAGE", "link": None},
            {"name": "receiver_mask", "type": "MASK", "link": None},
            {"name": "visibility", "type": "MASK", "link": None}
        ],
        "outputs": [
            {"name": "flash_transport", "type": "SL_FLASH", "links": [], "slot_index": 0},
            {"name": "irradiance_heatmap", "type": "IMAGE", "links": [], "slot_index": 1},
            {"name": "valid_geometry", "type": "MASK", "links": [], "slot_index": 2}
        ],
        "properties": {"Node name for S&R": "StudioTransportWall"},
        "widgets_values": ["encoded_x_right_y_up_z_toward_camera", 0.0, 0.0, 1.0, 40.0, 30.0, 20.0, 0.2],
        "title": "StudioLight · Cavity Wall Transport",
        "color": "#243141",
        "bgcolor": "#101d2d"
    }

    node_wall_prev = {
        "id": 56,
        "type": "PreviewImage",
        "pos": [450, 1250],
        "size": [360, 320],
        "flags": {},
        "order": 30,
        "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": None}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": [],
        "title": "Preview: Cavity Wall Irradiance"
    }

    # Add the new nodes
    new_nodes = [node_preflight, node_preflight_prev, node_repair, node_repair_prev, node_geometry, node_wall, node_wall_prev]
    v2["nodes"].extend(new_nodes)

    # 4. RE-WIRE LINKS
    # Find existing links from Node 12 (Draft VAEDecode):
    # Old link 23: 12 -> 14 (DSINE)
    # Old link 24: 12 -> 29 (Depth)
    # Old link 26: 12 -> 17 (Composite)
    # REROUTE THEM THROUGH NODE 52 (Repaired Draft)!
    
    # Connect Node 12 -> Node 50 (Preflight input 0 draft)
    l_12_to_preflight = current_link_id; current_link_id += 1
    v2["links"].append([l_12_to_preflight, 12, 0, 50, 0, "IMAGE"])
    node_preflight["inputs"][0]["link"] = l_12_to_preflight

    # Connect Node 12 -> Node 52 (Repair input 0 draft)
    l_12_to_repair = current_link_id; current_link_id += 1
    v2["links"].append([l_12_to_repair, 12, 0, 52, 0, "IMAGE"])
    node_repair["inputs"][0]["link"] = l_12_to_repair

    # Connect Node 14 (DSINE) -> Node 50 (Preflight input 1 normals)
    l_dsine_to_preflight = current_link_id; current_link_id += 1
    v2["links"].append([l_dsine_to_preflight, 14, 0, 50, 1, "IMAGE"])
    node_preflight["inputs"][1]["link"] = l_dsine_to_preflight

    # Connect Node 50 -> Node 51 (Preflight overlay preview)
    l_preflight_to_prev = current_link_id; current_link_id += 1
    v2["links"].append([l_preflight_to_prev, 50, 0, 51, 0, "IMAGE"])
    node_preflight_prev["inputs"][0]["link"] = l_preflight_to_prev

    # Connect Node 50 -> Node 52 (Preflight report -> Repair)
    l_report_to_repair = current_link_id; current_link_id += 1
    v2["links"].append([l_report_to_repair, 50, 1, 52, 1, "STRING"])
    node_repair["inputs"][1]["link"] = l_report_to_repair

    # Connect Node 52 -> Node 53 (Repair preview)
    l_repair_to_prev = current_link_id; current_link_id += 1
    v2["links"].append([l_repair_to_prev, 52, 0, 53, 0, "IMAGE"])
    node_repair_prev["inputs"][0]["link"] = l_repair_to_prev

    # REROUTE DSINE, Depth, and Composite to take Node 52 (Repaired Draft) instead of raw 12
    # Modify links in v2["links"]:
    for l in v2["links"]:
        lid, src_id, src_slot, dst_id, dst_slot, ltype = l
        if src_id == 12 and dst_id in [14, 29, 17]:
            # Change source to Node 52, slot 0
            l[1] = 52
            l[2] = 0

    # Wire Astra Wall Transport:
    # Node 29 (Depth) -> Node 54 (Geometry input 0 depth)
    l_depth_to_geom = current_link_id; current_link_id += 1
    v2["links"].append([l_depth_to_geom, 29, 0, 54, 0, "IMAGE"])
    node_geometry["inputs"][0]["link"] = l_depth_to_geom

    # Node 52 (Repair mask) -> Node 54 (Geometry input 1 subject_mask)
    l_mask_to_geom = current_link_id; current_link_id += 1
    v2["links"].append([l_mask_to_geom, 52, 1, 54, 1, "MASK"])
    node_geometry["inputs"][1]["link"] = l_mask_to_geom

    # Node 54 (Geometry) -> Node 55 (Wall input 0 geometry)
    l_geom_to_wall = current_link_id; current_link_id += 1
    v2["links"].append([l_geom_to_wall, 54, 0, 55, 0, "SL_GEOMETRY"])
    node_wall["inputs"][0]["link"] = l_geom_to_wall

    # Node 14 (DSINE normals) -> Node 55 (Wall input 1 normals)
    l_norm_to_wall = current_link_id; current_link_id += 1
    v2["links"].append([l_norm_to_wall, 14, 0, 55, 1, "IMAGE"])
    node_wall["inputs"][1]["link"] = l_norm_to_wall

    # Node 52 (Repair mask) -> Node 55 (Wall input 2 receiver_mask & input 3 visibility)
    l_mask_to_wall_rec = current_link_id; current_link_id += 1
    v2["links"].append([l_mask_to_wall_rec, 52, 1, 55, 2, "MASK"])
    node_wall["inputs"][2]["link"] = l_mask_to_wall_rec

    l_mask_to_wall_vis = current_link_id; current_link_id += 1
    v2["links"].append([l_mask_to_wall_vis, 52, 1, 55, 3, "MASK"])
    node_wall["inputs"][3]["link"] = l_mask_to_wall_vis

    # Node 55 -> Node 56 (Wall heatmap preview)
    l_wall_to_prev = current_link_id; current_link_id += 1
    v2["links"].append([l_wall_to_prev, 55, 1, 56, 0, "IMAGE"])
    node_wall_prev["inputs"][0]["link"] = l_wall_to_prev

    # Update Group titles & bounds to reflect the 5 Stages
    # Add Stage 2 & Stage 3 group
    v2["groups"].append({
        "id": 10,
        "title": "STAGE 2 & 3: PRE-FLIGHT GATE & APPEARANCE REPAIR",
        "bounding": [-430, -300, 420, 1260],
        "color": "#332211",
        "flags": {}
    })

    # Synchronize all node input and output link arrays with v2["links"]
    nodes_map = {n["id"]: n for n in v2["nodes"]}
    for n in nodes_map.values():
        for out in n.get("outputs", []):
            out["links"] = []
        for inp in n.get("inputs", []):
            inp["link"] = None

    for l in v2["links"]:
        lid, src_id, src_slot, dst_id, dst_slot, ltype = l
        if src_id in nodes_map:
            src_node = nodes_map[src_id]
            if src_slot < len(src_node.get("outputs", [])):
                out_links = src_node["outputs"][src_slot].get("links")
                if out_links is None:
                    src_node["outputs"][src_slot]["links"] = [lid]
                else:
                    out_links.append(lid)
        if dst_id in nodes_map:
            dst_node = nodes_map[dst_id]
            if dst_slot < len(dst_node.get("inputs", [])):
                dst_node["inputs"][dst_slot]["link"] = lid

    with open(OUTPUT_PATH, "w") as f:
        json.dump(v2, f, indent=2)

    print(f"✓ Successfully rebuilt StudioLight_v2_Rebuild.json at: {OUTPUT_PATH}")
    print(f" - Preserved StudioLightRig (Node 15) and VFX Radar (Node 16)!")
    print(f" - Preserved StudioPhotometricComposite (Node 17) and SIShadowHighlight (Node 18)!")
    print(f" - Inserted Stage 2 Preflight (Node 50) and Stage 3 Appearance Repair (Node 52)!")
    print(f" - Inserted Astra Cavity Wall Transport (Nodes 54, 55, 56)!")
    print(f" - Total nodes: {len(v2['nodes'])}, Total links: {len(v2['links'])}")

if __name__ == "__main__":
    build_v2()
