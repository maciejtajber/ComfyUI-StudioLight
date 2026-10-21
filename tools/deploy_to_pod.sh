#!/bin/bash
# One-command sync of StudioLight v2.0 to cloud GPU pod
if [ -z "$1" ]; then
    echo "Usage: ./deploy_to_pod.sh <IP:PORT>"
    exit 1
fi

CONN=$1
IP=$(echo $CONN | cut -d: -f1)
PORT=$(echo $CONN | cut -d: -f2)

echo "==> Deploying StudioLight v2.0 Adaptive Architecture to root@$IP:$PORT..."
LOCAL_NODES="/Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/nodes"
REMOTE_NODES="root@$IP:/app/ComfyUI/custom_nodes/ComfyUI-StudioLight/nodes"

scp -P $PORT $LOCAL_NODES/prompt_sanitizer.py $REMOTE_NODES/
scp -P $PORT $LOCAL_NODES/prompt_synthesizer.py $REMOTE_NODES/
scp -P $PORT $LOCAL_NODES/shader_engine.py $REMOTE_NODES/
scp -P $PORT $LOCAL_NODES/photometric_composite.py $REMOTE_NODES/
scp -P $PORT /Users/maciejtajber/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/StudioLight_Master.json root@$IP:/app/ComfyUI/custom_nodes/ComfyUI-StudioLight/workflows/

echo "==> Restarting ComfyUI on pod..."
ssh -p $PORT root@$IP 'pkill -f "main.py --port 8189" 2>/dev/null || true'
ssh -p $PORT root@$IP 'nohup /app/venv/bin/python /app/ComfyUI/main.py --listen 0.0.0.0 --port 8189 > /tmp/comfy.log 2>&1 &'
sleep 2

echo "==> Verifying ComfyUI status on pod..."
ssh -p $PORT root@$IP 'curl -s http://127.0.0.1:8189/system_stats' | grep -q "devices" && echo "✓ ComfyUI is ONLINE and healthy!" || echo "Check /tmp/comfy.log"
echo "==> Deployment Complete!"
