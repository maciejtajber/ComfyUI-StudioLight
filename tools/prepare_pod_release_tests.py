"""Install test-only helpers before starting/restarting the pod ComfyUI server."""
import argparse
from pathlib import Path
import shutil
p=argparse.ArgumentParser()
p.add_argument('--comfy-root',type=Path,required=True)
a=p.parse_args()
custom=a.comfy_root/'custom_nodes'
if not custom.is_dir(): p.error('ComfyUI custom_nodes directory does not exist')
target=custom/'StudioLightReleaseTests'
target.mkdir(exist_ok=True)
source=Path(__file__).with_name('release_test_nodes.py')
init=target/'__init__.py'
if init.exists() and init.read_bytes()!=source.read_bytes():
    raise SystemExit('Existing helper differs; preserve it and inspect before replacing')
shutil.copy2(source,init)
print(f'Installed {init}. Start/restart ComfyUI, then run tools/run_pod_release_tests.py.')
