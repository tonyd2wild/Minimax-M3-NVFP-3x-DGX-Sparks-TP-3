# MiniMax-M3 (NVFP4) on 3× DGX Spark — TP=3, 200K ctx, ~10.5 tok/s (RoCE)

> A working, verified recipe for serving `lukealonso/MiniMax-M3-NVFP4` (~243 GB, 428B-A23B MoE) at real tensor-parallel = 3 across three DGX Sparks, with clean tool-calling + reasoning (no `<mm:think>` / namespace-token leaks).

Built on **Luke Alonso's vLLM fork** (the `chthonic` build) + **b12x** kernels. This documents the parts that aren't in any existing guide: the **head-node OOM fixes** and the **multi-node Ray/NCCL setup** that took this from "crashes at warmup" to "serving + tool-calling clean."

## TL;DR

- **What you get:** MiniMax-M3-NVFP4 (428B-A23B MoE) serving at real TP=3 on 3× DGX Spark, tool-calling + reasoning clean.
- **Numbers:** 200K context; single-stream **~6 tok/s on the 1 GbE mgmt link**, **~10.5 tok/s on the 200 G RoCE mesh (+75%)**. Optional EAGLE3 speculative decoding stacks **~+25%** single-stream.
- **Status:** base TP=3 serving is verified, and the RoCE 200 G interconnect is **SOLVED** (2026-06-15).
- **Who it's for:** anyone reproducing multi-node TP=3 MoE serving on GB10 / sm_121 DGX Sparks.

## Hardware

- **3× NVIDIA DGX Spark** (GB10, **compute sm_121**, ARM64/aarch64, **128 GB unified memory** each).
- Nodes (mgmt net `10.0.0.0/24` over 1 GbE `enP7s7`): `bluey` <NODE0_IP> (head/rank0), `reddie` <NODE1_IP>, `asusi` <NODE2_IP>.
- Each Spark also has a **200 G ConnectX-7**; the 3 nodes form a **switchless point-to-point RoCE mesh** (see [RoCE 200 G interconnect](#roce-200-g-interconnect-solved-2026-06-15)).
- Model cached on all 3 at `~/.cache/huggingface/hub/models--lukealonso--MiniMax-M3-NVFP4`.

## Quick start

The full working image is published to GHCR — **no need to build chthonic yourself**:

```bash
docker pull ghcr.io/tonyd2wild/vllm-m3-chthonic:nccl230u1   # full NCCL/RoCE build (recommended)
# or: docker pull ghcr.io/tonyd2wild/vllm-m3-chthonic:latest   # base chthonic
```

Then run the multi-node launcher (`m3vllm.sh`, in this repo) **inside** that container across your 3× DGX Spark — head node + 2 workers. Start the **workers first** (they retry-join the head), then the leader:

```bash
# on reddie, then asusi:
docker start vllm_m3        # (or: docker run ... bash /m3vllm.sh worker)
# then on bluey (head):
docker start vllm_m3        # CMD = bash /m3vllm.sh leader
```

Bring-up is about 10 to 12 min (per-node 84 GB shard load + torch.compile + warmup + PIECEWISE cudagraph capture). A quiet stretch during safetensors load is normal — do not kill it.

Smoke test once `/v1/models` is up (see [Verify](#verify)):

```bash
bash verify-m3.sh        # runs on Bluey; proves generation + clean minimax_m3 tool-calling
```

**For agents:** point your agent at this repo; the image above + the scripts here are the complete, verified recipe.

## Setup (detailed)

### Weights

- HF model id: **`lukealonso/MiniMax-M3-NVFP4`** (~243 GB, 428B-A23B MoE).
- Cached on all 3 nodes at `~/.cache/huggingface/hub/models--lukealonso--MiniMax-M3-NVFP4` and mounted into the container at `/cache/huggingface`.

### Engine / image

- **Luke Alonso's vLLM fork** `local-inference-lab/vllm @ dev/chthonic-consecration` (build `0.11.2.dev279+chthonic.consecration.b12x.cu132`, torch 2.12.0+cu132, CUDA 13.2.1). Contains the MiniMax-M3 model code, the Rust `minimax_m3` tool/reasoning parsers, and commit **`fb63c9a` "Support MiniMax M3 TP3 virtual sharding"** (pads attention heads 64→96, KV 4→6 so the model shards cleanly ÷3 — **auto-applied at `--tensor-parallel-size 3`**, topology-agnostic).
- **b12x** kernel lib (`lukealonso/b12x`), installed **from master `08e980c`** at container start (the published PR build lacks `B12XPagedAttentionScratchCaps.copy_runtime_metadata` that chthonic needs).
- `cutlass-dsl` stays at the image's **4.5.2** (a runtime downgrade to 4.4.2 breaks flashinfer's `cutlass.cute.nvgpu.OperandMajorMode`; 4.5.2 compiles + warms clean on sm_121).

### Launch

The launcher (`m3vllm.sh`) runs **inside** the container; `leader` on the head, `worker` on the other two. The container is launched `--privileged --network host --ipc host --ulimit memlock=-1 --shm-size=32G`, with `-v ~/.cache/huggingface:/cache/huggingface` and `-v ~/m3vllm.sh:/m3vllm.sh:ro`, CMD `bash /m3vllm.sh leader|worker`.

```bash
#!/bin/bash
# M3 TP=3 multi-node vLLM launcher (runs INSIDE the vllm-m3-chthonic container).
# Usage: m3vllm.sh leader   (head, <NODE0_IP>)   |   m3vllm.sh worker   (the other 2)
set -x
ROLE="${1:?usage: m3vllm.sh leader|worker}"
HEAD_IP="${HEAD_IP:-<NODE0_IP>}"; RAY_PORT=6379; CLUSTER_GPUS=3

# b12x / M3 / arch envs (sm_121a for GB10)
export CUTE_DSL_ARCH=sm_121a
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1 VLLM_USE_AOT_COMPILE=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_MINIMAX_M3_MSA=1 VLLM_USE_B12X_SPARSE_INDEXER=1 VLLM_USE_B12X_FP8_GEMM=0
export VLLM_ENABLE_PCIE_ALLREDUCE=0          # multi-node: NCCL handles allreduce, not single-node PCIe
export SAFETENSORS_FAST_GPU=1
# NCCL over the 1GbE mgmt net (10.0.0.0/24 = the only common subnet across all 3 nodes; see Benchmarks)
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

Start the workers first, then the leader (see [Quick start](#quick-start)). The full launcher lives at [`m3vllm.sh`](m3vllm.sh); the RoCE data-path variant is [`m3vllm-roce.sh`](m3vllm-roce.sh).

### Verify

The whole point — tool-calling clean:

```bash
curl :8000/v1/chat/completions -d '{"model":"minimax-m3","messages":[{"role":"user",
  "content":"Weather in Seattle? use the tool"}],"tools":[{"type":"function","function":
  {"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
  "tool_choice":"auto","max_tokens":512}'
```

Expected: `finish_reason=tool_calls`, one clean `get_weather` call, `arguments` = valid JSON `{"city":"Seattle"}`, **zero** `<mm:think>` / `<tool_call>` / `<invoke>` leakage. Reasoning is returned under the `reasoning` key (not `reasoning_content`) — content carries the clean answer either way.

`verify-m3.sh` (run on Bluey against `localhost:8000`) automates this: `/v1/models`, a plain-chat reasoning-split check, and the tool-call check, each with a leak-grep.

## Benchmarks

- **On the 1 GbE management link: single-stream ~6 tok/s, ~10 tok/s aggregate @ 4 concurrent.** Modest, and the bottleneck there is the **interconnect, not compute**: enabling CUDA graphs only moved single-stream +0.3 tok/s, proving the time is spent waiting on the cross-node all-reduce. TP=3 does ~120 cross-node all-reduces per token; over 1 Gbps that dominates. (This is also why a PP=3 setup can feel comparable single-stream — PP only passes the hidden state twice per token.)
- **On the 200 G RoCE mesh: single-stream ~10.5 tok/s (+75% over the 1 GbE link).** Once on RoCE the single-stream path becomes **GPU-compute-bound**, so raw interconnect bandwidth beyond ~13 Gb/s does not raise single-stream further (**12.8 Gb/s and 111 Gb/s both give ~10.5 tok/s single-stream**). The full bandwidth pays off for **concurrency / aggregate throughput**, which at high context is bounded by KV-cache memory.
- **EAGLE3 speculative decoding stacks ~+25%** on top of RoCE on the single-stream path.

**EAGLE3 single stream (TP=3 over the 1 GbE management link):**

| config | single-stream | note |
|---|---|---|
| base M3 (no speculation) | ~6 tok/s | baseline |
| EAGLE3 + `enforce-eager` | ~6.3 tok/s | the eager penalty cancels the speculation gain |
| EAGLE3 + cudagraph (PIECEWISE) @ 128K | **~7.5 tok/s** | **about +25% over base** |

At PIECEWISE @ 128K: mean acceptance length ~2.6, draft accept rate ~55%, per-position ~0.73 / 0.55 / 0.34. Tool-calling stays clean (no token leak).

This is a living recipe. If you push past what we measured — higher single-stream or concurrent throughput on a switchless 3-Spark RoCE mesh, or the EAGLE3 draft beating +25% — please open an issue or PR.

## Configuration

Key tunable knobs (from the `vllm serve` command):

| flag | value | note |
|---|---|---|
| `--tensor-parallel-size` | `3` | real TP=3; `fb63c9a` virtual-sharding makes M3's 64-head / 4-KV-head attention divisible by 3 |
| `--max-model-len` | `200000` | drop to `128000` when EAGLE3 is on (`131072` fails by ~80 MB of KV; engine reported max feasible `129408`) |
| `--gpu-memory-utilization` | `0.82` | |
| `--kv-cache-dtype` | `fp8_e4m3` | |
| `--max-num-seqs` | `2` | concurrency cap |
| `--max-num-batched-tokens` | `512` | with `--enable-chunked-prefill` |
| `--block-size` | `128` | KV page size |
| `--quantization` | `modelopt_fp4` | NVFP4 target quant |
| `--attention-backend` / `--moe-backend` | `B12X_ATTN` / `b12x` | b12x kernels |

Two optional speed levers, each with its exact, verified state:

### EAGLE3 speculative decoding (~+25% single-stream, stacks on RoCE)

MiniMax-M3 ships **no native MTP / speculative weights** (the `MiniMaxM3MTP` architecture exists in the model code but ships **zero trained weights**), so we drive an **external EAGLE3 draft**: [`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3), a 1-layer `LlamaForCausalLMEagle3` (num_attention_heads 64, hidden_size 6144, head_dim 128). The draft must be padded 64→96 heads for TP=3 — see [Troubleshooting → EAGLE3 bring-up](#eagle3-bring-up-the-4-walls) and `pad_eagle3_draft.py`; we built the result as **`MiniMax-M3-EAGLE3-pad96`**.

Working speculative-config (add to the `vllm serve` command):

```bash
--speculative-config '{"model": "/path/to/MiniMax-M3-EAGLE3-pad96", "method": "eagle3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "attention_backend": "TRITON_ATTN"}'
```

**Note:** with EAGLE3 on, the KV cache is slightly tighter, so drop `--max-model-len` from **200000 to 128000**. (A `131072` attempt failed by ~80 MB of KV; the engine reported a max feasible of `129408`.)

**Key insight:** the draft correctly predicts ~2.6 tokens per step, but on the **1 GbE link** that does **not** become a ~2.4x speedup, because TP=3 is communication-bound there (roughly 120 small all-reduces per token). Speculation gives you the tokens; the slow interconnect eats the savings. That is exactly why the **interconnect (RoCE) was the first lever to pull**. EAGLE3's **+25% stacks on top of RoCE** on the single-stream path.

### RoCE 200 G interconnect

Move NCCL / vLLM TP=3 model traffic off the 1 GbE management link and onto the **200 G ConnectX-7 RoCE mesh**. The 3 nodes form a **switchless point-to-point mesh** (no switch), each leg on its own **/30 subnet** (192.168.100 / 101 / 102), **RoCEv2 GID index 3**. Serve with the `nccl230u1` image and `m3vllm-roce.sh`. The RoCE NCCL env block (in `m3vllm-roce.sh`):

```bash
NCCL_IB_DISABLE=0
NCCL_NET=IB
NCCL_SOCKET_IFNAME=enP7s7              # bootstrap/control on the 1GbE mgmt NIC (OOB); DATA rides RoCE
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1      # each node's two slot-1 RoCE ports reach its two neighbors
NCCL_IB_ADDR_RANGE=192.168.100.0/22
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_MERGE_NICS=0
NCCL_NET_PLUGIN=none
NCCL_CROSS_NIC=1                       # asymmetric mesh: each rocep1s0f0 faces a different neighbor
NCCL_NET_GDR_LEVEL=LOC                 # disable GPUDirect RDMA (GB10 unified-memory safety)
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_IB_ROCE_VERSION_NUM=2
# Do NOT hardcode NCCL_IB_GID_INDEX. Leave it default so dynamic per-peer GID selection works WITH
# NCCL_IB_ADDR_RANGE. Hardcoding GID_INDEX=3 disables that and re-introduces the err-110 cross-pairing.
```

The leave-GID-default point matters: `NCCL_IB_ADDR_RANGE` only steers GID selection when the GID index is **unset**. That dynamic, CIDR-driven selection is how subnet-aware-routing pairs the right local HCA to each neighbor on the switchless /30 mesh. (Underneath, the mesh is RoCEv2 at GID index 3, but you let NCCL find it rather than pin it.) The two non-obvious bring-up fixes it took to get here are in [Troubleshooting → RoCE bring-up](#roce-200-g-interconnect-solved-2026-06-15).

## Troubleshooting

### Base bring-up: the 5 fixes that actually mattered

Everything else is "follow Luke's flags." These are the non-obvious ones that gate a working bring-up:

1. **`--load-format safetensors`** — `instanttensor`'s GDS `open()` throws `_C.open std::exception` under torch 2.12 on Spark (no GPUDirect Storage). Use plain safetensors.
2. **`--object-store-memory 1073741824`** (1 GB) on **every** `ray start` — Ray defaults to reserving ~30 % of RAM (~36 GB/node) for a plasma object store **vLLM TP never uses** (tensors go over NCCL). On the head (which alone also runs the Ray GCS + API server) that reserve + the 84 GB shard + KV **overcommits the 121 GB box → `dmesg: NVRM: Out of memory` during weight-load → rank-0 dies.** Capping it freed ~35 GB/node (cluster object store 109 GB → 3 GB) and killed the load-time OOM.
3. **`RAY_memory_monitor_refresh_ms=0`** — after a *fully successful* warmup the head sits at ~96 % RAM (normal for a loaded model on unified memory). Ray's generic **95 % memory monitor** then **false-kills** the TP0 worker (`exit_type=NODE_OUT_OF_MEMORY`, classified `OOMContext`) — but there is **no real OOM** (no NVRM error, no Linux OOM-kill, ~4.4 GB still free). Disable the monitor; the Linux kernel + NVIDIA driver remain the real backstops. (Ray's own log recommends this exact var.)
4. **b12x from master** (`08e980c`, pure-python) — see [Engine / image](#engine--image); the PR wheel is missing `copy_runtime_metadata`.
5. **`fb63c9a` virtual-TP sharding** — nothing to set; it engages automatically at TP=3 and is what makes M3's 64-head / 4-KV-head attention divisible by 3. Without this commit, TP=3 is impossible (you'd be stuck on PP).

**Diagnostic tip that saved hours:** we don't run with `--rm`, so a crashed container keeps `/tmp/ray`. `docker cp vllm_m3:/tmp/ray /tmp/...` then read `raylet`'s `threshold_memory_monitor.cc` / `node_manager.cc` kill message for the exact memory numbers + the `RAY_memory_...` env hint. `dmesg | grep NVRM` distinguishes a **real** driver OOM from Ray's heuristic false-kill.

### EAGLE3 bring-up: the 4 walls

Getting the EAGLE3 draft to load on **TP=3** meant clearing **four distinct walls, in order**:

1. **`SpeculativeConfig` divisibility error** → set **`draft_tensor_parallel_size: 1`**. Run the draft on a single GPU instead of splitting it across all 3 ranks.
2. **Quantization error (`hf_overrides must be a dict`)** → **omit `quantization` from the speculative-config** so the draft loads in **bf16**. The draft is tiny, there is no need to quantize it (and trying to apply the target's NVFP4 quant to it is what dead-ends).
3. **Draft construction assert (num_heads not divisible by TP=3) in `llama_eagle3.py`** → **pad the draft from 64 to 96 attention heads.** 96 is the **only** valid target, because it must satisfy **both**:
   - transformers config validation: `hidden_size % num_heads == 0` → `6144 / 96 == 64` (OK), and
   - TP divisibility: `96 / 3 == 32` (OK).
   The naive head_dim-based guess of **66 heads FAILS** transformers config validation: `"hidden size (6144) is not a multiple of the number of attention heads (66)"`. The padding zero-fills the q/k/v_proj **out-features 8192 → 12288** and the o_proj **in-features 8192 → 12288**, in **bf16**, and leaves every other tensor untouched. See **`pad_eagle3_draft.py`** in this repo; we built the result as **`MiniMax-M3-EAGLE3-pad96`**. (`validate_eagle3_pad.py` and `validate_eagle3_real_tp3.py` dry-check the padded shapes + run the real vLLM TP=3 weight loaders against them.)
4. **Draft attention backend.** The draft inherits the engine's **fp8 KV cache** (`--kv-cache-dtype fp8_e4m3`) **and block-size 128**. With that combo: `FLASH_ATTN` is rejected (`kv_cache_dtype not supported`), and `FLASHINFER` is rejected at block-size 128 (`page size >= 128 requires trtllm-gen attention`). **`TRITON_ATTN`** is the one backend valid for `(head_dim 128, fp8 KV, block 128)` on **sm_121**, confirmed by running vLLM's own backend validator. So set **`"attention_backend": "TRITON_ATTN"`** in the speculative-config.

### RoCE 200 G interconnect (SOLVED 2026-06-15)

Two fixes cracked the switchless RoCE mesh; the second is the genuinely non-obvious, community-useful finding.

**Fix 1 — NCCL version (build v2.30u1 from source for sm_121).** The gating env is **`NCCL_IB_SUBNET_AWARE_ROUTING`**, which makes NCCL map each rank-pair to the correct cable on the switchless mesh. It was **introduced in NCCL 2.30**. Luke Alonso's container shipped **NCCL 2.29.7**, which lacks it. So we built **NVIDIA NCCL tag `v2.30u1` from source** for **sm_121** and committed a new image tag, **`vllm-m3-chthonic:nccl230u1`**. Build (see `nccl230-build.sh` + `nccl230-inner-build.sh`, which build inside a throwaway container and `docker commit` the new tag; `nccl-build-watcher.sh` polls all 3 nodes for DONE/FAIL):

```bash
git clone --depth 1 -b v2.30u1 https://github.com/NVIDIA/nccl.git
cd nccl && make -j src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"
# installs to /opt/nccl230/build/lib inside the committed image
```

**Verify the runtime version with ctypes `ncclGetVersion`, NOT `torch.cuda.nccl.version()`** (the latter reports the **compile-time constant** and is misleading).

**Fix 2 — the baked LD_PRELOAD shim (the non-obvious one).** Building 2.30u1 and symlinking it in **was not enough**: the runtime NCCL banner kept reading **2.30.4**, and the **`ibv_modify_qp` err 110** cross-pairing persisted, even though 2.30u1 was installed on disk. The cause: the vLLM container had a **baked `LD_PRELOAD`** pointing at an **old local-inference NCCL shim** (`libnccl-local-inference.so.2.30.4`). A baked `LD_PRELOAD` **overrides both a symlink swap AND an `LD_LIBRARY_PATH` prepend**, so the container silently kept running the 2.30.4 shim. That shim lacked the working subnet-aware device-override, which is exactly what produced the persistent err-110 cross-pairing on the switchless mesh (the err-110 was a symptom of the wrong library, not a wiring fault). The fix, applied in the RoCE launcher **after** the `pip install` steps (which themselves reinstall the nvidia-nccl wheel and clobber the symlink, so this must run last):

```bash
export LD_PRELOAD=/opt/nccl230/build/lib/libnccl.so.2
unset VLLM_NCCL_SO_PATH NCCL_LOCAL_INFERENCE_PATH NCCL_PR2127_PATH
export LD_LIBRARY_PATH=/opt/nccl230/build/lib:$LD_LIBRARY_PATH
```

Then verify: the launcher should print **`FORCED_NCCL_VERSION 23007`** and the NCCL banner should read **`NCCL version 2.30.7`**. With the right library actually loaded, NCCL logs **`Connected all rings`** over **NET/IB** with **zero err-110**.

**Bandwidth gotcha — a COLD POWER-DRAIN, not a warm reboot (credit: mashie).** Raw `ib_write_bw` was initially stuck at **~12.8 Gb/s** and **did not scale with queue-pair count**, on otherwise-healthy **Gen5 x4 / 200 G** hardware. Forum user **mashie** had the answer: a **cold power-drain** clears a stuck NIC/PHY state. Gracefully shut down all 3 Sparks, **unplug the power bricks for 60-90 seconds**, then power back on. After the drain, `ib_write_bw` jumped to **111.85 Gb/s** (full line rate, matching eugr's reference). **A warm reboot does NOT clear it; only pulling power does.**

**Reboot gotcha:** a full power-cycle (or any reboot) reverts the ConnectX-7 **runtime MTU 9000 → 1500**. Re-apply the jumbo MTU with `ip link set <ifname> mtu 9000` after boot (the netplan `/30` mesh and the committed NCCL image persist; only the runtime MTU resets). Also note: the **first vLLM boot after a power-drain loads shards slower** (cold page cache) and can transiently stall mid-load — a plain restart (stop all + relaunch worker-first) clears it; it is not a regression.

**RoCE results:** TP=3 MiniMax-M3-NVFP4 serving over RoCE at 200K context, tool-calling clean. Single-stream ~10.5 tok/s vs ~6 tok/s on the 1 GbE mgmt link (+75%), which matches the old PP=3. Honest finding: once on RoCE, single-stream is **GPU-compute-bound**, so interconnect bandwidth beyond ~13 Gb/s does **not** raise single-stream — **12.8 Gb/s and 111 Gb/s both give ~10.5 tok/s single-stream.** The full 111 Gb/s pays off for **concurrency / aggregate throughput** (more simultaneous requests), which at high context is bounded by **KV-cache memory**, not the link.

## Credits & links

- **Luke Alonso** — `local-inference-lab/vllm` chthonic fork + `b12x` + the `MiniMax-M3-NVFP4` quant + the `fb63c9a` TP3 virtual-sharding commit.
- **eugr** (`eugr/spark-vllm-docker`, `docs/NETWORKING.md`) — the core switchless 3-Spark RoCE mesh recipe.
- **mashie** (NVIDIA DGX Spark forum) — the **cold-power-drain** tip that took raw `ib_write_bw` from ~12.8 Gb/s to 111.85 Gb/s.
- A **ChatGPT-assisted debugging pass** that isolated the baked-`LD_PRELOAD` local-inference NCCL shim.
- **Inferact** — the [`MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3) draft (EAGLE3 speculative decoding).
- The **NVIDIA DGX Spark forum** community (thread *"MiniMax M3 NVFP4 for quad DGX Spark"*).

Recipe assembled, OOM/Ray fixes diagnosed, and EAGLE3 + RoCE (SOLVED) completed on a live 3-Spark cluster, 2026-06-15.
