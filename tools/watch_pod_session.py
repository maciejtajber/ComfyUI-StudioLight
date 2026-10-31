#!/usr/bin/env python3
"""
StudioLight Pod Live Watcher & Session Mirror
Continuously mirrors all raw outputs and intermediate passes from the cloud GPU pod to the local Mac.
"""
import os
import sys
import time
import subprocess
from pathlib import Path

POD_IP = sys.argv[1] if len(sys.argv) > 1 else "159.48.242.3"
POD_PORT = sys.argv[2] if len(sys.argv) > 2 else "21402"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

LOCAL_DEST = Path(os.path.expanduser("~/Desktop/StudioLight_Pod_Session"))
OUTPUTS_DIR = LOCAL_DEST / "output"
TEMP_DIR = LOCAL_DEST / "temp"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(f"🎬 STUDIOLIGHT POD LIVE WATCHER ACTIVE")
print(f"📍 Remote Pod: {POD_IP}:{POD_PORT}")
print(f"📁 Local Mac Destination: {LOCAL_DEST}")
print(f"   • Full Outputs: {OUTPUTS_DIR}")
print(f"   • Mid Passes & Previews: {TEMP_DIR}")
print("=" * 65)
print("Watching for new renders and intermediate passes every 4s... (Press Ctrl+C to stop)\n")

known_outputs = set(os.listdir(OUTPUTS_DIR))
known_temps = set(os.listdir(TEMP_DIR))

ssh_opts = f"ssh -o StrictHostKeyChecking=no -p {POD_PORT} -i {SSH_KEY}"

try:
    while True:
        # 1. Sync final outputs
        res_out = subprocess.run(
            ["rsync", "-az", "-e", ssh_opts, f"root@{POD_IP}:/app/ComfyUI/output/", f"{OUTPUTS_DIR}/"],
            capture_output=True, text=True
        )
        current_outputs = set(os.listdir(OUTPUTS_DIR))
        new_outputs = current_outputs - known_outputs
        for f in sorted(new_outputs):
            if not f.startswith("."):
                print(f"✨ [NEW OUTPUT] {f} -> {OUTPUTS_DIR.name}/{f}")
        known_outputs = current_outputs

        # 2. Sync temp mid outputs
        res_temp = subprocess.run(
            ["rsync", "-az", "-e", ssh_opts, f"root@{POD_IP}:/app/ComfyUI/temp/", f"{TEMP_DIR}/"],
            capture_output=True, text=True
        )
        current_temps = set(os.listdir(TEMP_DIR))
        new_temps = current_temps - known_temps
        for f in sorted(new_temps):
            if not f.startswith("."):
                print(f"🔍 [MID PASS]   {f} -> {TEMP_DIR.name}/{f}")
        known_temps = current_temps

        time.sleep(4)
except KeyboardInterrupt:
    print("\n🛑 Watcher stopped by user.")
