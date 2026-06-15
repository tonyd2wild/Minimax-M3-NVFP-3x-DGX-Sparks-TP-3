# MiniMax-M3 (NVFP4) serving at Tensor-Parallel = 3 across 3× DGX Spark (GB10 / sm_121)

A **working, verified** recipe for running `lukealonso/MiniMax-M3-NVFP4` (~243 GB, 428B-A23B MoE)
on **three DGX Sparks at real TP=3** with **clean tool-calling + reasoning** (no `<mm:think>` /
namespace-token leaks). Built on **Luke Alonso's vLLM fork** (the `chthonic` build) + **b12x** kernels.

This documents the parts that aren't in any existing guide: the **head-node OOM fixes** and the
**multi-node Ray/NCCL setup** that took this from "crashes at warmup" to "serving + tool-calling clean."

> Status: **base TP=3 serving is verified.** Single-stream throughput is modest (~6 tok/s) and the
> bottleneck is the inter-node interconnect, not compute - see [Performance](#performance--whats-next).
> Two speed levers (RoCE, EAGLE3) are documented at the bottom with their exact current state.

---

## Hardware / topology

- **3× NVIDIA DGX Spark** (GB10, **compute sm_121**, ARM64/aarch64, **128 GB unified memory** each).
- Nodes (mgmt net `10.0.0.0/24` over 1 GbE `enP7s7`): `bluey` 10.0.0.6 (head/rank0), `reddie` 10.0.0.9, `asusi` 10.0.0.5.
- Model cached on all 3 at `~/.cache/huggingface/hub/models--lukealonso--MiniMax-M3-NVFP4`.

## Engine / image

- **Luke Alonso's vLLM fork** `local-inference-lab/vllm @ dev/chthonic-consecration`
  (build `0.11.2.dev279+chthonic.consecration.b12x.cu132`, torch 2.12.0+cu132, CUDA 13.2.1).
  Contains the MiniMax-M3 model code, the Rust `minimax_m3` tool/reasoning parsers, and
  commit **`fb63c9a` "Support MiniMax M3 TP3 virtual sharding"** (pads attention heads 64→96,
  KV 4→6 so the model shards cleanly ÷3 - **auto-applied at `--tensor-parallel-size 3`**, topology-agnostic).
- **b12x** kernel lib (`lukealonso/b12x`), installed **from master `08e980c`** at container start
  (the published PR build lacks `B12XPagedAttentionScratchCaps.copy_runtime_metadata` that chthonic needs).
- `cutlass-dsl` stays at the image's **4.5.2** (a runtime downgrade to 4.4.2 breaks flashinfer's
  `cutlass.cute.nvgpu.OperandMajorMode`; 4.5.2 compiles + warms clean on sm_121).

## The 5 fixes that actually mattered

Everything else is "follow Luke's flags." These are the non-obvious ones that gate a working bring-up:

1. **`--load-format safetensors`** - `instanttensor`'s GDS `open()` throws `_C.open std::exception`
   under torch 2.12 on Spark (no GPUDirect Storage). Use plain safetensors.
2. **`--object-store-memory 1073741824`** (1 GB) on **every** `ray start` - Ray defaults to reserving
   ~30 % of RAM (~36 GB/node) for a plasma object store **vLLM TP never uses** (tensors go over NCCL).
   On the head (which alone also runs the Ray GCS + API server) that reserve + the 84 GB shard + KV
   **overcommits the 121 GB box → `dmesg: NVRM: Out of memory` during weight-load → rank-0 dies.**
   Capping it freed ~35 GB/node (cluster object store 109 GB → 3 GB) and killed the load-time OOM.
3. **`RAY_memory_monitor_refresh_ms=0`** - after a *fully successful* warmup the head sits at ~96 %
   RAM (normal for a loaded model on unified memory). Ray's generic **95 % memory monitor** then
   **false-kills** the TP0 worker (`exit_type=NODE_OUT_OF_MEMORY`, classified `OOMContext`) - but there
   is **no real OOM** (no NVRM error, no Linux OOM-kill, ~4.4 GB still free). Disable the monitor;
   the Linux kernel + NVIDIA driver remain the real backstops. (Ray's own log recommends this exact var.)
4. **b12x from master** (`08e980c`, pure-python) - see above; the PR wheel is missing `copy_runtime_metadata`.
5. **`fb63c9a` virtual-TP sharding** - nothing to set; it engages automatically at TP=3 and is what makes
   M3's 64-head / 4-KV-head attention divisible by 3. Without this commit, TP=3 is impossible (you'd be stuck on PP).

Diagnostic tip that saved hours: we don't run with `--rm`, so a crashed container keeps `/tmp/ray`.
`docker cp vllm_m3:/tmp/ray /tmp/...` then read `raylet`'s `threshold_memory_monitor.cc` / `node_manager.cc`
kill message for the exact memory numbers + the `RAY_memory_...` env hint. `dmesg | grep NVRM`
distinguishes a **real** driver OOM from Ray's heuristic false-kill.

## The launcher (`m3vllm.sh`)

Runs **inside** the container; `leader` on the head, `worker` on the other two. Container is launched
`--privileged --network host --ipc host --ulimit memlock=-1 --shm-size=32G`, with
`-v ~/.cache/huggingface:/cache/huggingface` and `-v ~/m3vllm.sh:/m3vllm.sh:ro`, CMD `bash /m3vllm.sh leader|worker`.

```bash
#!/bin/bash
# M3 TP=3 multi-node vLLM launcher (runs INSIDE the vllm-m3-chthonic container).
# Usage: m3vllm.sh leader   (head, 10.0.0.6)   |   m3vllm.sh worker   (the other 2)
set -x
ROLE="${1:?usage: m3vllm.sh leader|worker}"
HEAD_IP="${HEAD_IP:-10.0.0.6}"; RAY_PORT=6379; CLUSTER_GPUS=3

# b12x / M3 / arch envs (sm_121a for GB10)
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_MINIMAX_M3_MSA=1 VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_B12X_FP8_GEMM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=0          # multi-node: NCCL handles allreduce, not single-node PCIe
export SAFETENSORS_FAST_GPU=1
# NCCL over the 1GbE mgmt net (10.0.0.0/24 = the only common subnet across all 3 nodes; see Performance)
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=enP7s7 GLOO_SOCKET_IFNAME=enP7s7
export NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1 NCCL_DEBUG=WARN
export HF_HOME=/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0       # FIX #3: disable Ray's 95% false-kill watchdog

SELF_IP="$(hostname -I | tr ' ' '\n' | grep -E '^10\.0\.0\.' | head -1)"

# ray + b12x-master are not in the image; install at start (cached across restarts)
python -c "import ray" 2>/dev/null || pip install -q "ray==2.55.1"
python -c "import inspect; from b12x.integration.paged_attention_scratch import B12XPagedAttentionScratchCaps as C; raise SystemExit(0 if 'copy_runtime_metadata' in inspect.signature(C.__init__).parameters else 1)" 2>/dev/null \
  || pip install -q --force-reinstall --no-deps git+https://github.com/lukealonso/b12x.git@08e980c303b0b6291700a6b85aa09aa874fc27cb

if [ "$ROLE" = "worker" ]; then
  for i in $(seq 1 60); do
    ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=1 --node-ip-address="$SELF_IP" \
      --object-store-memory=1073741824 --block && exit 0   # FIX #2: cap plasma store
    echo "worker: head not ready, retry 5s"; sleep 5
  done; exit 1
fi

# leader
ray start --head --port="${RAY_PORT}" --num-gpus=1 --node-ip-address="${HEAD_IP}" \
  --dashboard-host=0.0.0.0 --object-store-memory=1073741824   # FIX #2
for i in $(seq 1 90); do ray status 2>/dev/null | grep -qE "/${CLUSTER_GPUS}\.0 GPU" && break; sleep 5; done

exec vllm serve lukealonso/MiniMax-M3-NVFP4 \
  --served-model-name minimax-m3 --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 3 --distributed-executor-backend ray \
  --gpu-memory-utilization 0.82 \
  --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --attention-backend B12X_ATTN --moe-backend b12x \
  -cc.mode=VLLM_COMPILE -cc.cudagraph_mode=PIECEWISE \
  --block-size 128 --load-format safetensors \          # FIX #1
  --max-model-len 200000 --max-num-seqs 2 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --enable-prefix-caching \
  --skip-mm-profiling --mm-encoder-tp-mode data \        # vision tower 16 heads ∤3 -> replicate
  --reasoning-parser minimax_m3 --enable-auto-tool-choice --tool-call-parser minimax_m3
```

## Launch procedure (worker-first)

Start the **workers first** (they retry-join the head), then the leader:

```
# on reddie, then asusi:
docker start vllm_m3        # (or: docker run ... bash /m3vllm.sh worker)
# then on bluey (head):
docker start vllm_m3        # CMD = bash /m3vllm.sh leader
```

Bring-up ≈ 10–12 min (per-node 84 GB shard load + torch.compile + warmup + PIECEWISE cudagraph capture).
A quiet stretch during safetensors load is normal - do not kill it.

## Verification (the whole point - tool-calling clean)

```
curl :8000/v1/chat/completions -d '{"model":"minimax-m3","messages":[{"role":"user",
  "content":"Weather in Seattle? use the tool"}],"tools":[{"type":"function","function":
  {"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
  "tool_choice":"auto","max_tokens":512}'
```
Expected: `finish_reason=tool_calls`, one clean `get_weather` call, `arguments` = valid JSON
`{"city":"Seattle"}`, **zero** `<mm:think>` / `<tool_call>` / `<invoke>` leakage. Reasoning is returned
under the `reasoning` key (not `reasoning_content`) - content carries the clean answer either way.

## Performance + what's next

- **Single-stream ~6 tok/s, ~10 tok/s aggregate @ 4 concurrent.** Modest - and the bottleneck is the
  **interconnect, not compute**: enabling CUDA graphs only moved single-stream +0.3 tok/s, proving the
  time is spent waiting on the cross-node all-reduce.
- **NCCL is running over the 1 GbE management NIC (`enP7s7`).** TP=3 does ~120 cross-node all-reduces per
  token; over 1 Gbps that dominates. (This is also why a PP=3 setup can feel faster single-stream - PP
  only passes the hidden state twice per token.) The 200 G ConnectX-7 ports sit unused for model traffic.
- **RoCE over the 200 G ring (the real fix, in progress):** the 3 Sparks cable into a triangle
  (two QSFP ports each, no switch). Naively pointing NCCL at RoCE **fails** because a single global
  `NCCL_IB_GID_INDEX` can't describe point-to-point /30 links and NCCL pairs the wrong HCA→peer
  (`ibv_modify_qp ... 110 Connection timed out`). Working approach: give each leg a static /30, remove
  zeroconf 169.254 addrs so the RoCE-v2 GID is at a consistent index, then **unset `NCCL_IB_GID_INDEX`**
  and let NCCL pick per-connection via `NCCL_IB_ADDR_RANGE=<fabric CIDR>` + `NCCL_IB_ADDR_FAMILY=AF_INET`,
  with `NCCL_IB_HCA=<cabled HCAs only>`, `NCCL_CROSS_NIC=1`, and **`NCCL_NET_GDR_LEVEL=0`** (mandatory on
  GB10 - unified memory breaks GPU-Direct RDMA). NVIDIA flags switchless-3-node NCCL as a manual-build
  rough edge; a small RoCE switch (one common /24) is the fully-supported fallback.
- **EAGLE3 speculative decoding (M3 has no native MTP):** the chthonic M3 class implements `SupportsEagle3`,
  and `Inferact/MiniMax-M3-EAGLE3` loads - set `draft_tensor_parallel_size:1` (the draft's 64 heads aren't
  ÷3 and don't get virtual-TP padding). Current blocker: the bf16 draft has no quant config, so vLLM tries
  to apply the **NVFP4** target quant to it and `get_quant_config` dead-ends (`hf_overrides must be a dict`).
  Open: ship the draft a small fp8 quant config, or wait for a pre-quantized eagle3 draft.

## Credits

Luke Alonso (`local-inference-lab/vllm` chthonic fork + `b12x` + the `MiniMax-M3-NVFP4` quant + the
`fb63c9a` TP3 virtual-sharding commit). eugr's Spark vLLM work. The NVIDIA DGX Spark forum community
(thread *"MiniMax M3 NVFP4 for quad DGX Spark"*). Recipe assembled + the OOM/Ray fixes diagnosed on a
live 3-Spark cluster, 2026-06-15.
