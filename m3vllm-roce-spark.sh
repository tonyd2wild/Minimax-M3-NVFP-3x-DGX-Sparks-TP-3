#!/bin/bash
# M3 TP=3 multi-node vLLM launcher, RoCE 200G data path — adapted for our 3-node DGX Spark ring.
# Runs INSIDE the vllm-m3-chthonic:nccl230u1 container (NCCL 2.30.7 w/ subnet-aware-routing).
# Usage: m3vllm-roce-spark.sh leader   (head 192.168.86.48)  |  m3vllm-roce-spark.sh worker (worker1/2)
set -x
ROLE="${1:?usage: m3vllm-roce-spark.sh leader|worker}"
HEAD_IP="${HEAD_IP:-192.168.86.48}"
RAY_PORT=6379
CLUSTER_GPUS=3

# --- b12x / M3 / arch envs (sm_121a for GB10) ---
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_MINIMAX_M3_MSA=1 VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_B12X_FP8_GEMM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=1
export B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export SAFETENSORS_FAST_GPU=1

# --- NCCL over RoCE (eugr 3-node mesh recipe). NCCL 2.30.7 (v2.30u1) provides SUBNET_AWARE_ROUTING. ---
# Control/bootstrap rides the common 192.168.86.x subnet; DATA plane rides 100% IB/RoCE (192.168.100.0/20).
# Expose all 4 logical HCA endpoints (Issue #1 fix) to achieve full ~183 Gbps bandwidth across GB10 PCIe-x4 halves.
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=wlP9s9 GLOO_SOCKET_IFNAME=wlP9s9
export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
# CRITICAL: do NOT hardcode NCCL_IB_GID_INDEX. Leaving it at default lets NCCL dynamically
# select the correct RoCEv2 GID per-peer by CIDR via NCCL_IB_ADDR_RANGE.
export NCCL_IB_ADDR_RANGE=192.168.100.0/20
export NCCL_IB_MERGE_NICS=0
export NCCL_NET_PLUGIN=none
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_CROSS_NIC=1
export NCCL_IB_ADDR_FAMILY=AF_INET
export NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV
export HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0

# Extract self IP from mgmt net (192.168.86.x)
SELF_IP="$(hostname -I | tr ' ' '\n' | grep -E '^192\.168\.86\.' | head -1)"
echo "ROLE=$ROLE SELF_IP=$SELF_IP HEAD_IP=$HEAD_IP"

# Ray (not in Luke's single-node image).
if ! python -c "import ray" 2>/dev/null; then
  pip install -q "ray==2.55.1" 2>&1 | tail -3 || pip install -q "ray[default]==2.55.1" 2>&1 | tail -3
fi

# b12x master (copy_runtime_metadata for chthonic).
if ! python -c "import inspect; from b12x.integration.paged_attention_scratch import B12XPagedAttentionScratchCaps as C; raise SystemExit(0 if 'copy_runtime_metadata' in inspect.signature(C.__init__).parameters else 1)" 2>/dev/null; then
  echo "Upgrading b12x -> master (08e980c)..."
  pip install -q --force-reinstall --no-deps git+https://github.com/lukealonso/b12x.git@08e980c303b0b6291700a6b85aa09aa874fc27cb 2>&1 | tail -3
fi

# Force the host-built NCCL v2.30u1 to actually load. The ray/b12x pip installs above reinstall the
# nvidia-nccl wheel (2.30.4) and clobber our symlink, so re-point after pip installs.
PIPNCCL=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib
if [ -e /opt/nccl230/build/lib/libnccl.so.2 ]; then
  rm -f "$PIPNCCL/libnccl.so.2" 2>/dev/null
  ln -sf /opt/nccl230/build/lib/libnccl.so.2 "$PIPNCCL/libnccl.so.2"
  unset VLLM_NCCL_SO_PATH NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH
  export LD_PRELOAD=/opt/nccl230/build/lib/libnccl.so.2
  export LD_LIBRARY_PATH=/opt/nccl230/build/lib:${LD_LIBRARY_PATH}
  python -c "import ctypes;l=ctypes.CDLL('/opt/nccl230/build/lib/libnccl.so.2');v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v));print('FORCED_NCCL_VERSION',v.value)" 2>/dev/null || true
fi

sync; echo 1 > /proc/sys/vm/drop_caches 2>/dev/null || true

if [ "$ROLE" = "worker" ]; then
  for i in $(seq 1 60); do
    if ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=1 --node-ip-address="$SELF_IP" \
         --object-store-memory=1073741824 --block; then
      exit 0
    fi
    echo "worker: head not ready, retry in 5s..."; sleep 5
  done
  echo "worker: timed out joining head"; exit 1
fi

# leader
ray start --head --port="${RAY_PORT}" --num-gpus=1 --node-ip-address="$HEAD_IP" --dashboard-host=0.0.0.0 \
  --object-store-memory=1073741824
echo "WAIT_FOR_${CLUSTER_GPUS}_GPU"
for i in $(seq 1 90); do
  if ray status 2>/dev/null | grep -qE "/${CLUSTER_GPUS}\.0 GPU"; then echo "RAY_CLUSTER_FULL_${CLUSTER_GPUS}_GPU"; break; fi
  sleep 5
done
ray status 2>&1 | tail -20

exec vllm serve /cache/huggingface/hub/models--lukealonso--MiniMax-M3-NVFP4 \
  --served-model-name minimax-m3 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 3 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.82 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend B12X_ATTN \
  --moe-backend b12x \
  -cc.mode=VLLM_COMPILE \
  -cc.cudagraph_mode=PIECEWISE \
  --cudagraph-capture-sizes 1 2 \
  --block-size 128 \
  --load-format safetensors \
  --max-model-len 200000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --mm-encoder-tp-mode data \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
