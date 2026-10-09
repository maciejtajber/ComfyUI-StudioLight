#!/usr/bin/env python3
"""
ComfyUI-StudioLight — Hardcore Stress-Testing Suite for 3D Camera-Space Metric Shadows
Author: Maciej Tajber (https://github.com/maciejtajber) // METAMATERIAL
License: MIT

Executes 4 rigorous mathematical stress tests on the metric shadow engine:
1. Oblique Grazing Ray & Wedge Test (Steep Angles, Staircasing, Acne Margin ε)
2. Slit Aperture Banding vs. IGN Dither Penumbra Analysis (Reproducing User Plates)
3. Extreme Depth Chasm Stress (2m -> 5m -> 10m -> 24m Projective Scaling)
4. Hardware Benchmark & VRAM Saturation Matrix (512², 1024², 2048² on RTX 3090)
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Ensure custom nodes directory is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from nodes.metric_shadow import (
    CameraModel,
    EmitterSpec,
    MODIFIER_EMITTERS,
    generate_emitter_samples_3d,
    metric_shadow_visibility_area,
    metric_shadow_visibility_points,
    metric_shadow_visibility_point,
    project_camera_points,
)

OUTPUT_DIR = Path(os.environ.get("STUDIOLIGHT_STRESS_OUT", "/tmp/studiolight_stress"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

print(f"=== ComfyUI-StudioLight Hardcore Stress Test Suite ===")
print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"Output Directory: {OUTPUT_DIR}\n")


def create_camera_field(
    h: int, w: int, camera: CameraModel, depth_map: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstructs camera-space surface_xyz from a given depth_map."""
    py = torch.arange(h, device=DEVICE, dtype=DTYPE)
    px = torch.arange(w, device=DEVICE, dtype=DTYPE)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")
    
    # NDC in [-1, 1]
    u = (grid_x / (w - 1.0)) * 2.0 - 1.0
    v = (grid_y / (h - 1.0)) * 2.0 - 1.0
    if camera.image_y_down:
        v = -v
        
    tan_half = camera.tan_half_fov
    aspect = camera.aspect
    
    # In OpenGL camera space, -Z is positive depth
    z = -depth_map
    x = u * depth_map * (tan_half * aspect)
    y = v * depth_map * tan_half
    
    surface_xyz = torch.stack((x, y, z), dim=-1).unsqueeze(0)  # [1, H, W, 3]
    scene_depth = depth_map.unsqueeze(0)                       # [1, H, W]
    return surface_xyz, scene_depth


def save_image_tensor(tensor_hw: torch.Tensor, path: Path):
    """Saves a [H, W] float tensor [0, 1] to PNG."""
    arr = torch.clamp(tensor_hw, 0.0, 1.0).detach().cpu().numpy()
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(path)


# ==============================================================================
# TEST 1: Oblique Grazing Ray & Wedge Test
# ==============================================================================
def run_test_1_grazing_wedge():
    print("--> Running Test 1: Oblique Grazing Ray & Wedge Test...")
    h, w = 512, 512
    cam = CameraModel(vertical_fov_deg=40.0, aspect=1.0)
    
    # Construct a sloped terrain plane (slanted away along Y)
    py = torch.linspace(0, 1, h, device=DEVICE, dtype=DTYPE)[:, None]
    px = torch.linspace(0, 1, w, device=DEVICE, dtype=DTYPE)[None, :]
    
    # Base plane depth slopes steeply from 1.8m to 3.8m
    depth = 1.8 + 2.0 * py.expand(h, w)
    
    # Add a prominent elevated wedge/block in the center
    wedge_mask = (py >= 0.35) & (py <= 0.65) & (px >= 0.25) & (px <= 0.70)
    # The wedge steps closer by 0.5m
    depth = torch.where(wedge_mask, depth - 0.50, depth)
    
    surface_xyz, scene_depth = create_camera_field(h, w, cam, depth)
    
    # Place a hard point light at an extreme grazing angle (theta > 80°)
    # Light positioned high to the right, skimming the slope
    light_pos = torch.tensor([[1.8, 1.6, -1.5]], device=DEVICE, dtype=DTYPE)
    
    results = {}
    bias_values = [0.001, 0.003, 0.006, 0.012]
    vis_panels = []
    
    for bias in bias_values:
        vis = metric_shadow_visibility_point(
            surface_xyz,
            scene_depth,
            light_pos,
            cam,
            steps=24,
            depth_bias_m=bias,
            ray_start_m=0.012,
            dither=False,
        )[0]
        
        # Unoccluded region (top left of base slope)
        slope_free_region = vis[0:100, 0:100]
        self_shadow_rate = float((slope_free_region < 0.5).to(torch.float32).mean().item()) * 100.0
        
        # Occluded shadow region behind the wedge
        shadow_core = vis[380:480, 200:400]
        shadow_darkness = float((shadow_core < 0.1).to(torch.float32).mean().item()) * 100.0
        
        results[f"bias_{bias}m"] = {
            "self_shadow_acne_pct": self_shadow_rate,
            "shadow_coverage_pct": shadow_darkness,
        }
        vis_panels.append(vis)
        print(f"    Bias {bias:.3f}m -> Self-shadow acne: {self_shadow_rate:.2f}%, Shadow density: {shadow_darkness:.2f}%")
        
    # Stitch panels into a comparative strip: [Bias 0.001, 0.003, 0.006, 0.012]
    strip = torch.cat(vis_panels, dim=1)
    save_image_tensor(strip, OUTPUT_DIR / "stress_01_wedge_grazing_rays.png")
    
    print(f"    ✓ Test 1 complete. Saved {OUTPUT_DIR / 'stress_01_wedge_grazing_rays.png'}")
    return results


# ==============================================================================
# TEST 2: Slit Aperture Banding vs. IGN Dither Penumbra Analysis
# ==============================================================================
def run_test_2_slit_banding_vs_dither():
    print("--> Running Test 2: Slit Aperture Banding vs. IGN Dither (Reproducing User Plates)...")
    h, w = 512, 512
    cam = CameraModel(vertical_fov_deg=40.0, aspect=1.0)
    
    # Flat background wall at Z = -4.0m
    depth = torch.full((h, w), 4.0, device=DEVICE, dtype=DTYPE)
    
    # Slit / Tall rectangular pillar at Z = -2.0m (Delta Z = 2.0m chasm)
    # Centered horizontally, tall aspect ratio like user's plate
    slit_mask = (torch.abs(torch.linspace(-1, 1, h, device=DEVICE)[:, None]) <= 0.60) & \
                (torch.abs(torch.linspace(-1, 1, w, device=DEVICE)[None, :]) <= 0.20)
    depth = torch.where(slit_mask, 2.0, depth)
    
    surface_xyz, scene_depth = create_camera_field(h, w, cam, depth)
    
    # Large Octabox (1.4m) offset to the left and slightly up
    octabox = MODIFIER_EMITTERS["Octabox (Soft)"]
    light_pos = torch.tensor([[-0.8, 0.3, -1.0]], device=DEVICE, dtype=DTYPE)
    light_aim = torch.tensor([[0.0, 0.0, -2.0]], device=DEVICE, dtype=DTYPE)
    
    # Ground Truth Reference: S=64, steps=64, no dither
    print("    Computing 64-sample Quasi-Monte Carlo Ground Truth...")
    ref_vis = metric_shadow_visibility_area(
        surface_xyz, scene_depth, light_pos, light_aim, octabox, cam,
        num_emitter_samples=64, ray_steps=64, sample_chunk=8, dither=False
    )[0]
    
    configs = [
        {"name": "A_S4_step8_nodither",  "S": 4,  "steps": 8,  "dither": False},
        {"name": "B_S8_step16_nodither", "S": 8,  "steps": 16, "dither": False},
        {"name": "C_S4_step8_dither",    "S": 4,  "steps": 8,  "dither": True},
        {"name": "D_S8_step16_dither",   "S": 8,  "steps": 16, "dither": True},
        {"name": "E_S16_step24_dither",  "S": 16, "steps": 24, "dither": True},
    ]
    
    panels = []
    panel_names = []
    metrics = {}
    
    # Extract 1D cross-section row through the penumbra (row 256)
    row_idx = 256
    ref_row = ref_vis[row_idx].detach().cpu().numpy()
    cross_sections = {"GroundTruth_S64": ref_row.tolist()}
    
    for cfg in configs:
        vis = metric_shadow_visibility_area(
            surface_xyz, scene_depth, light_pos, light_aim, octabox, cam,
            num_emitter_samples=cfg["S"], ray_steps=cfg["steps"], sample_chunk=4, dither=cfg["dither"]
        )[0]
        
        # Calculate error metrics against Reference
        mae = float(torch.abs(vis - ref_vis).mean().item())
        mse = float(F.mse_loss(vis, ref_vis).item())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        
        # Quantify 1D cross-section roughness / step count
        row = vis[row_idx].detach().cpu().numpy()
        cross_sections[cfg["name"]] = row.tolist()
        
        # Discrete step detection (derivative flat-spots)
        penumbra_slice = row[280:420]
        diffs = np.abs(np.diff(penumbra_slice))
        # Total variation
        tv = float(np.sum(diffs))
        
        metrics[cfg["name"]] = {
            "S": cfg["S"],
            "steps": cfg["steps"],
            "dither": cfg["dither"],
            "MAE": mae,
            "PSNR_dB": psnr,
            "TotalVariation": tv,
        }
        panels.append(vis)
        panel_names.append(cfg["name"])
        print(f"    Config {cfg['name']}: MAE={mae:.4f}, PSNR={psnr:.2f}dB, TV={tv:.4f}")
        
    panels.append(ref_vis)
    panel_names.append("Ref_S64_GT")
    
    # Save 2x3 comparison montage
    top_row = torch.cat(panels[:3], dim=1)
    bottom_row = torch.cat(panels[3:], dim=1)
    montage = torch.cat([top_row, bottom_row], dim=0)
    save_image_tensor(montage, OUTPUT_DIR / "stress_02_slit_banding_vs_dither.png")
    
    # Export cross sections for chart presentation
    with open(OUTPUT_DIR / "stress_02_cross_sections.json", "w") as f:
        json.dump(cross_sections, f)
        
    print(f"    ✓ Test 2 complete. Saved {OUTPUT_DIR / 'stress_02_slit_banding_vs_dither.png'}")
    return metrics


# ==============================================================================
# TEST 3: Extreme Depth Chasm Stress (2m -> 5m -> 10m -> 24m)
# ==============================================================================
def run_test_3_extreme_depth_chasm():
    print("--> Running Test 3: Extreme Depth Chasm Stress (Projective Scaling)...")
    h, w = 512, 512
    cam = CameraModel(vertical_fov_deg=40.0, aspect=1.0)
    
    # Foreground occluding disk at Z = -1.5m
    py = torch.linspace(-1, 1, h, device=DEVICE)[:, None]
    px = torch.linspace(-1, 1, w, device=DEVICE)[None, :]
    sphere_mask = (px**2 + py**2) <= 0.08  # radius ~0.28m
    
    chasms = [
        {"name": "Chasm_2m",  "wall_z": 3.5,  "delta_z": 2.0},
        {"name": "Chasm_5m",  "wall_z": 6.5,  "delta_z": 5.0},
        {"name": "Chasm_10m", "wall_z": 11.5, "delta_z": 10.0},
        {"name": "Chasm_24m", "wall_z": 25.5, "delta_z": 24.0},
    ]
    
    octabox = MODIFIER_EMITTERS["Octabox (Soft)"]
    light_pos = torch.tensor([[0.8, 0.8, -0.5]], device=DEVICE, dtype=DTYPE)
    light_aim = torch.tensor([[0.0, 0.0, -1.5]], device=DEVICE, dtype=DTYPE)
    
    panels = []
    chasm_metrics = {}
    
    for ch in chasms:
        depth = torch.full((h, w), ch["wall_z"], device=DEVICE, dtype=DTYPE)
        depth = torch.where(sphere_mask, 1.5, depth)
        
        surface_xyz, scene_depth = create_camera_field(h, w, cam, depth)
        
        vis = metric_shadow_visibility_area(
            surface_xyz, scene_depth, light_pos, light_aim, octabox, cam,
            num_emitter_samples=8, ray_steps=24, sample_chunk=4, dither=True
        )[0]
        
        # Measure penumbra transition width on background wall
        # Profile line slicing through cast shadow
        row = vis[256].detach().cpu().numpy()
        shadow_pixels = np.where(row < 0.90)[0]
        if len(shadow_pixels) > 0:
            shadow_diameter = float(shadow_pixels[-1] - shadow_pixels[0])
        else:
            shadow_diameter = 0.0
            
        penumbra_pixels = np.where((row > 0.10) & (row < 0.90))[0]
        penumbra_width = float(len(penumbra_pixels))
        
        chasm_metrics[ch["name"]] = {
            "delta_z_meters": ch["delta_z"],
            "wall_depth_meters": ch["wall_z"],
            "shadow_diameter_px": shadow_diameter,
            "penumbra_width_px": penumbra_width,
        }
        panels.append(vis)
        print(f"    {ch['name']} (ΔZ={ch['delta_z']}m) -> Shadow Diam: {shadow_diameter}px, Penumbra Width: {penumbra_width}px")
        
    strip = torch.cat(panels, dim=1)
    save_image_tensor(strip, OUTPUT_DIR / "stress_03_extreme_depth_chasm.png")
    
    print(f"    ✓ Test 3 complete. Saved {OUTPUT_DIR / 'stress_03_extreme_depth_chasm.png'}")
    return chasm_metrics


# ==============================================================================
# TEST 4: Hardware Benchmark & VRAM Saturation Matrix on RTX 3090
# ==============================================================================
def run_test_4_rtx3090_benchmark():
    print("--> Running Test 4: RTX 3090 Hardware Benchmark & VRAM Saturation...")
    cam = CameraModel(vertical_fov_deg=40.0, aspect=1.0)
    octabox = MODIFIER_EMITTERS["Octabox (Soft)"]
    light_pos = torch.tensor([[0.8, 0.8, -0.5]], device=DEVICE, dtype=DTYPE)
    light_aim = torch.tensor([[0.0, 0.0, -2.0]], device=DEVICE, dtype=DTYPE)
    
    resolutions = [512, 1024, 2048]
    presets = [
        {"name": "Fast_Preview", "S": 4,  "steps": 12, "chunk": 4},
        {"name": "Balanced",     "S": 8,  "steps": 16, "chunk": 4},
        {"name": "High_End",     "S": 16, "steps": 24, "chunk": 4},
    ]
    
    benchmark_data = {}
    
    for res in resolutions:
        depth = torch.full((res, res), 3.0, device=DEVICE, dtype=DTYPE)
        # Disk occluder
        py = torch.linspace(-1, 1, res, device=DEVICE)[:, None]
        px = torch.linspace(-1, 1, res, device=DEVICE)[None, :]
        depth = torch.where((px**2 + py**2) <= 0.15, 1.8, depth)
        
        surface_xyz, scene_depth = create_camera_field(res, res, cam, depth)
        
        benchmark_data[f"{res}x{res}"] = {}
        print(f"    --- Resolution {res}x{res} ---")
        
        for p in presets:
            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                
            # Warm-up run
            _ = metric_shadow_visibility_area(
                surface_xyz, scene_depth, light_pos, light_aim, octabox, cam,
                num_emitter_samples=p["S"], ray_steps=p["steps"], sample_chunk=p["chunk"], dither=True
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
                
            # Timed iterations
            times_ms = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = metric_shadow_visibility_area(
                    surface_xyz, scene_depth, light_pos, light_aim, octabox, cam,
                    num_emitter_samples=p["S"], ray_steps=p["steps"], sample_chunk=p["chunk"], dither=True
                )
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
                
            avg_time_ms = float(np.mean(times_ms))
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if DEVICE.type == "cuda" else 0.0
            
            # Rays evaluated per second
            total_rays = res * res * p["S"] * p["steps"]
            rps_millions = (total_rays / (avg_time_ms / 1000.0)) / 1e6
            
            benchmark_data[f"{res}x{res}"][p["name"]] = {
                "S": p["S"],
                "steps": p["steps"],
                "chunk": p["chunk"],
                "latency_ms": round(avg_time_ms, 2),
                "peak_vram_mb": round(peak_vram_mb, 2),
                "mega_ray_steps_per_sec": round(rps_millions, 2),
            }
            print(f"      [{p['name']}]: {avg_time_ms:.2f} ms | VRAM: {peak_vram_mb:.1f} MB | Throughput: {rps_millions:.2f} M steps/sec")
            
    print(f"    ✓ Test 4 complete.")
    return benchmark_data


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    start_total = time.perf_counter()
    
    t1_results = run_test_1_grazing_wedge()
    t2_results = run_test_2_slit_banding_vs_dither()
    t3_results = run_test_3_extreme_depth_chasm()
    t4_results = run_test_4_rtx3090_benchmark()
    
    total_elapsed = time.perf_counter() - start_total
    
    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_suite_elapsed_sec": round(total_elapsed, 2),
        "test_1_grazing_wedge": t1_results,
        "test_2_slit_banding_vs_dither": t2_results,
        "test_3_extreme_depth_chasm": t3_results,
        "test_4_rtx3090_benchmark": t4_results,
    }
    
    report_path = OUTPUT_DIR / "studiolight_hardcore_stress_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
        
    print(f"\n========================================================")
    print(f"🎉 All 4 Hardcore Stress Tests Completed in {total_elapsed:.2f}s!")
    print(f"Full Report: {report_path}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
