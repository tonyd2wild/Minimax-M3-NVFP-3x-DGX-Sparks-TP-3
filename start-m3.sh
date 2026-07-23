#!/bin/bash
# start-m3.sh — Start the MiniMax-M3 TP=3 cluster across 3 DGX Spark nodes.
# Run from the head node (gx10-0d82 / 192.168.86.48).
set -e

HEAD_MGMT=192.168.86.48
HEAD_ROCE=192.168.100.1
WORKER1_MGMT=192.168.86.49
WORKER2_MGMT=192.168.86.45
IMAGE=ghcr.io/tonyd2wild/vllm-m3-chthonic:nccl230u1
CONTAINER=vllm_m3

echo "============================================"
echo "  MiniMax-M3 TP=3 Cluster Start (RoCE Only)"
echo "============================================"
echo

# --- Stop any existing containers first ---
echo "[1/4] Cleaning up existing containers..."
for NODE in "$WORKER1_MGMT" "$WORKER2_MGMT"; do
  echo "  -> $NODE: stopping $CONTAINER..."
  ssh "spark@$NODE" "sg docker -c 'docker stop $CONTAINER 2>/dev/null; docker rm $CONTAINER 2>/dev/null' || true" 2>&1 | tail -1
done
echo "  -> head: stopping $CONTAINER..."
sg docker -c "docker stop $CONTAINER 2>/dev/null; docker rm $CONTAINER 2>/dev/null" || true
echo

# --- Sync launcher script to workers (so the bind mount finds a file, not a dir) ---
echo "[0/4] Syncing launcher script to workers..."
scp /home/spark/repo/m3-recipe/m3vllm-roce-spark.sh "spark@$WORKER1_MGMT:/home/spark/m3vllm-roce-spark.sh" 2>&1 | tail -1
sshpass -p 'Aslk1234!' scp /home/spark/repo/m3-recipe/m3vllm-roce-spark.sh "spark@$WORKER2_MGMT:/home/spark/m3vllm-roce-spark.sh" 2>&1 | tail -1
echo

# --- Start workers first (they retry-join the head) ---
echo "[2/4] Starting WORKER1 ($WORKER1_MGMT)..."
ssh "spark@$WORKER1_MGMT" "sg docker -c 'docker run -d --name $CONTAINER --privileged --network host --ipc host \
  --gpus all --ulimit memlock=-1 --shm-size=32G \
  --entrypoint bash \
  -v /home/spark/.cache/huggingface:/cache/huggingface \
  -v /home/spark/m3vllm-roce-spark.sh:/m3vllm.sh:ro \
  -v /home/spark/patches/qwen3_dflash.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro \
  -e HEAD_IP=$HEAD_MGMT \
  $IMAGE \
  /m3vllm.sh worker'" 2>&1
echo "  Worker1 started."
echo

echo "[3/4] Starting WORKER2 ($WORKER2_MGMT)..."
ssh "spark@$WORKER2_MGMT" "sg docker -c 'docker run -d --name $CONTAINER --privileged --network host --ipc host \
  --gpus all --ulimit memlock=-1 --shm-size=32G \
  --entrypoint bash \
  -v /home/spark/.cache/huggingface:/cache/huggingface \
  -v /home/spark/m3vllm-roce-spark.sh:/m3vllm.sh:ro \
  -v /home/spark/patches/qwen3_dflash.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro \
  -e HEAD_IP=$HEAD_MGMT \
  $IMAGE \
  /m3vllm.sh worker'" 2>&1
echo "  Worker2 started."
echo

# --- Start leader on head ---
echo "[4/4] Starting HEAD/LEADER ($HEAD_MGMT)..."
sg docker -c "docker run -d --name $CONTAINER --privileged --network host --ipc host \
  --gpus all --ulimit memlock=-1 --shm-size=32G \
  --entrypoint bash \
  -v /home/spark/.cache/huggingface:/cache/huggingface \
  -v /home/spark/repo/m3-recipe/m3vllm-roce-spark.sh:/m3vllm.sh:ro \
  -v /home/spark/repo/m3-recipe/patches/qwen3_dflash.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_dflash.py:ro \
  -e HEAD_IP=$HEAD_MGMT \
  $IMAGE \
  /m3vllm.sh leader" 2>&1
echo "  Head/leader started."
echo

# --- Start OpenCode reasoning translation proxy ---
echo "[5/4] Starting OpenCode reasoning translation proxy on port 8001..."
# Prefer the systemd user service (auto-starts on boot, auto-restarts on crash).
# Fall back to the legacy tmux session if systemd is unavailable.
if systemctl --user list-unit-files m3-reasoning-proxy.service >/dev/null 2>&1 \
   && systemctl --user restart m3-reasoning-proxy.service 2>/dev/null; then
  echo "  Proxy started (systemd: m3-reasoning-proxy.service)."
else
  tmux kill-session -t m3_proxy 2>/dev/null || true
  tmux new-session -d -s m3_proxy "python3 /home/spark/repo/m3-recipe/m3_reasoning_proxy.py" 2>/dev/null || true
  echo "  Proxy started (legacy tmux: m3_proxy session)."
fi
echo

echo "============================================"
echo "  M3 cluster starting."
echo "  Bring-up takes ~10-12 min (shard load + JIT compile + cudagraph)."
echo "  Check logs:  sg docker -c 'docker logs -f $CONTAINER'"
echo "  Status:      /home/spark/repo/m3-recipe/status-m3.sh"
echo "  Verify:      curl http://$HEAD_MGMT:8000/v1/models"
echo "============================================"
