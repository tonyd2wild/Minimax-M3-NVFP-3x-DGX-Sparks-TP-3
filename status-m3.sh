#!/bin/bash
# status-m3.sh — Show status of the MiniMax-M3 TP=3 cluster across all 3 DGX Spark nodes.
# Run from the head node (gx10-0d82 / 192.168.86.48).
set +e

CONTAINER=vllm_m3
HEAD_MGMT=192.168.86.48

echo "============================================"
echo "  MiniMax-M3 TP=3 Cluster Status"
echo "============================================"
echo

echo "=== HEAD (192.168.86.48) ==="
echo "--- Container ---"
sg docker -c "docker ps -a --filter name=$CONTAINER --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" 2>&1
echo "--- Last 5 log lines ---"
sg docker -c "docker logs --tail 5 $CONTAINER" 2>&1
echo

echo "=== WORKER1 (192.168.86.49) ==="
echo "--- Container ---"
ssh spark@192.168.86.49 "sg docker -c 'docker ps -a --filter name=$CONTAINER --format \"table {{.Names}}\t{{.Status}}\t{{.Ports}}\"'" 2>&1
echo "--- Last 5 log lines ---"
ssh spark@192.168.86.49 "sg docker -c 'docker logs --tail 5 $CONTAINER'" 2>&1
echo

echo "=== WORKER2 (192.168.86.45) ==="
echo "--- Container ---"
ssh spark@192.168.86.45 "sg docker -c 'docker ps -a --filter name=$CONTAINER --format \"table {{.Names}}\t{{.Status}}\t{{.Ports}}\"'" 2>&1
echo "--- Last 5 log lines ---"
ssh spark@192.168.86.45 "sg docker -c 'docker logs --tail 5 $CONTAINER'" 2>&1
echo

echo "=== vLLM Endpoint Check ==="
echo "--- curl http://$HEAD_MGMT:8000/v1/models ---"
curl -s --connect-timeout 5 "http://$HEAD_MGMT:8000/v1/models" 2>&1 || echo "  Not responding yet."
echo

echo "============================================"
echo "  RoCE Ring IPs:"
echo "    Head port0:    192.168.100.1/30 <-> Worker1 port1: 192.168.100.2/30"
echo "    Worker1 port0: 192.168.101.1/30 <-> Worker2 port1: 192.168.101.2/30"
echo "    Worker2 port0: 192.168.102.1/30 <-> Head port1:    192.168.102.2/30"
echo "============================================"
