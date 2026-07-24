#!/bin/bash
# stop-m3.sh — Stop the MiniMax-M3 TP=3 cluster across all 3 DGX Spark nodes.
# Run from the head node (gx10-0d82 / 192.168.86.48).
set +e

CONTAINER=vllm_m3

echo "============================================"
echo "  MiniMax-M3 TP=3 Cluster Stop"
echo "============================================"
echo

echo "[1/3] Stopping HEAD..."
sg docker -c "docker stop $CONTAINER 2>/dev/null; docker rm $CONTAINER 2>/dev/null" || true
echo "  Head stopped."
echo

echo "[2/3] Stopping WORKER1 (192.168.86.49)..."
ssh spark@192.168.86.49 "sg docker -c 'docker stop $CONTAINER 2>/dev/null; docker rm $CONTAINER 2>/dev/null' || true" 2>&1
echo "  Worker1 stopped."
echo

echo "[3/3] Stopping WORKER2 (192.168.86.45)..."
ssh spark@192.168.86.45 "sg docker -c 'docker stop $CONTAINER 2>/dev/null; docker rm $CONTAINER 2>/dev/null' || true" 2>&1
echo "  Worker2 stopped."
echo

# --- Stop OpenCode reasoning translation proxy ---
echo "[4/3] Stopping OpenCode reasoning translation proxy..."
# Stop the systemd user service if present; also kill any legacy tmux session.
systemctl --user stop m3-reasoning-proxy.service 2>/dev/null || true
tmux kill-session -t m3_proxy 2>/dev/null || true
echo "  Proxy stopped."
echo

echo "============================================"
echo "  M3 cluster stopped on all nodes."
echo "============================================"
