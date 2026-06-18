#!/bin/bash
# M3 TP=3 multi-node vLLM launcher (runs INSIDE the vllm-m3-chthonic container).
# Usage: m3vllm.sh leader    (on Bluey/head, <NODE0_IP>)
#        m3vllm.sh worker    (on Reddie/Asusi)
set -x
ROLE="${1:?usage: m3vllm.sh leader|worker}"
HEAD_IP="${HEAD_IP:-<NODE0_IP>}"
RAY_PORT=6379
CLUSTER_GPUS=3

# --- b12x / M3 / arch envs (from Luke's serve-minimax-m3-nvfp4.sh, sm_121a for GB10) ---
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_MINIMAX_M3_MSA=1 VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_B12X_FP8_GEMM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=0            # multi-node: NCCL handles allreduce, not single-node PCIe
export VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=1
export B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=1
export TORCH_SHOW_CPP_STACKTRACES=1
# CUDA_LAUNCH_BLOCKING diagnostic complete (2026-06-15): proved rank-0 death is NOT a CUDA kernel fault
# (zero synchronous CUDA errors after full warmup). Root cause = head-node unified-memory overcommit. Keep OFF.
export CUDA_LAUNCH_BLOCKING=0
export SAFETENSORS_FAST_GPU=1
# --- NCCL on the 1GbE mgmt net (enP7s7 / 10.0.0.0/24 = the ONLY common subnet across all 3 nodes) ---
# RoCE-ring attempt 2026-06-15 got FURTHER than sglang (bootstrap + IB channels formed over NET/IB) but the
# data-plane QP connects TIME OUT (ibv_modify_qp err 110): on our SWITCHLESS point-to-point /30 cabling NCCL
# pairs the WRONG HCA->peer (e.g. Bluey rocep1s0f0=192.168.101.1/Asusi-link trying to reach Reddie 192.168.100.2
# = a different cable). Switchless RoCE needs a RoCE switch (common L2 subnet) OR the NVIDIA Sync-Cluster-Assistant
# NCCL-2.30u1 topology config - NOT a quick env fix. 3rd /30 (192.168.102) is configured + pingable for a future retry.
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=enP7s7 GLOO_SOCKET_IFNAME=enP7s7
export NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_DEBUG=WARN
export HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RAY_DEDUP_LOGS=0
# Disable Ray's memory-monitor worker-killer (2026-06-15): on this unified-memory box the node legitimately
# sits at ~96% during inference (the model is SUPPOSED to use most of RAM); Ray's generic 95% threshold is a
# false alarm and killed TP0 right after a FULLY SUCCESSFUL warmup (no real OOM: no NVRM, no Linux OOM-kill,
# 4.4GB still free). Ray's own log recommends this exact var. Linux/NVRM remain the real OOM backstop.
export RAY_memory_monitor_refresh_ms=0

SELF_IP="$(hostname -I | tr ' ' '\n' | grep -E '^10\.0\.0\.' | head -1)"
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

# NOTE: cutlass-dsl stays at the image's 4.5.2 - a runtime downgrade to eugr's 4.4.2 breaks flashinfer
# (needs 4.5.x cutlass.cute.nvgpu.OperandMajorMode). 4.4.2 would require a FULL rebuild pinned to 4.4.2.
# Our 4.5.2 compiles+warms clean (no ptxas reject), so the cutlass PTX bug (#3227) is likely already fixed in 4.5.2.

# safetensors loader (instanttensor GDS open() throws under torch 2.12 on Spark). Drop page cache pre-load.
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
# --object-store-memory cap (2026-06-15): RESTORES Luke's recipe value (1GB) that our hand-rolled 3-node
# launcher had dropped -> Ray was defaulting to ~36GB/node (confirmed: cluster showed 109GB = 36x3).
# On the HEAD that reserve + 84GB shard + KV overcommits the 121GB box -> dmesg "NVRM: Out of memory"
# -> rank-0 dies post-warmup. Cap frees ~35GB/node. vLLM TP never uses the store (tensors go over NCCL).
# Root cause CONFIRMED 2026-06-15: Ray logs show "death context type = OOMContext / task failed due to oom"
# for the TP0 worker on the head -> head-node memory overcommit (not a CUDA/kernel fault).
# (No --temp-dir: only /cache/huggingface is mounted; the container keeps /tmp/ray on stop since we don't --rm,
#  so raylet logs are recoverable via `docker cp vllm_m3:/tmp/ray ...` for post-mortem.)
ray start --head --port="${RAY_PORT}" --num-gpus=1 --node-ip-address="${HEAD_IP}" --dashboard-host=0.0.0.0 \
  --object-store-memory=1073741824
echo "WAIT_FOR_${CLUSTER_GPUS}_GPU"
for i in $(seq 1 90); do
  if ray status 2>/dev/null | grep -qE "/${CLUSTER_GPUS}\.0 GPU"; then echo "RAY_CLUSTER_FULL_${CLUSTER_GPUS}_GPU"; break; fi
  sleep 5
done
ray status 2>&1 | tail -20

# Phase 2 (OPTIONAL) EAGLE3 speculative decoding (~+25% single-stream over base).
# M3 ships NO trained MTP weights, so we drive an external EAGLE3 draft. To enable:
#   1. build the padded draft once: python pad_eagle3_draft.py  (64 -> 96 heads, see README Phase 2)
#   2. drop --max-model-len from 200000 to 128000 (the draft KV makes 131072 fail by ~80MB)
#   3. add the --speculative-config line shown below to the vllm serve args.
# SPEC_CFG='{"model": "/cache/huggingface/MiniMax-M3-EAGLE3-pad96", "method": "eagle3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "attention_backend": "TRITON_ATTN"}'

exec vllm serve lukealonso/MiniMax-M3-NVFP4 \
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
  --block-size 128 \
  --load-format safetensors \
  --max-model-len 200000 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 512 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --mm-encoder-tp-mode data \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3
  # Phase 2: append the next line (and set --max-model-len 128000 above) to turn on EAGLE3:
  #   --speculative-config "$SPEC_CFG"
