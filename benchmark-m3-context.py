#!/usr/bin/env python3
"""
Context Latency & Throughput Profiler for MiniMax-M3
Measures TTFT (Time To First Token) and generation speed (tok/s) across scaling prompt token lengths.
"""
import time
import json
import argparse
import urllib.request

def generate_synthetic_prompt(approx_tokens: int) -> str:
    # Approx 4 characters per token
    target_chars = approx_tokens * 4
    base_text = "The quick brown fox jumps over the lazy dog. MiniMax M3 NVFP4 serving on DGX Spark. "
    repeats = (target_chars // len(base_text)) + 1
    return (base_text * repeats)[:target_chars]

def run_benchmark_step(endpoint: str, model_name: str, approx_tokens: int, max_output_tokens: int):
    prompt_text = generate_synthetic_prompt(approx_tokens)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Respond with a concise summary."},
            {"role": "user", "content": f"Here is the context:\n{prompt_text}\n\nPlease summarize key points."}
        ],
        "max_tokens": max_output_tokens,
        "stream": True
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith("data: "):
                    data_body = line_str[6:].strip()
                    if data_body == "[DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.time()
                    token_count += 1
    except Exception as e:
        return {"error": str(e), "tokens": approx_tokens}
    
    end_time = time.time()
    ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0
    decode_duration = (end_time - first_token_time) if first_token_time else (end_time - start_time)
    tok_per_sec = (token_count / decode_duration) if decode_duration > 0 else 0.0
    
    return {
        "tokens": approx_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "output_tokens": token_count,
        "decode_sec": round(decode_duration, 2),
        "tok_per_sec": round(tok_per_sec, 2),
        "error": None
    }

def main():
    parser = argparse.ArgumentParser(description="MiniMax-M3 Context Latency Benchmark")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8009/v1/chat/completions", help="API Endpoint URL")
    parser.add_argument("--model", default="minimax-m3", help="Model name")
    parser.add_argument("--steps", nargs="+", type=int, default=[5000, 20000, 50000, 100000, 150000, 180000], help="Context token lengths to benchmark")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per step")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"MiniMax-M3 Context Latency & Throughput Benchmark")
    print(f"Target Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print("=" * 70)
    print(f"{'Target Context':<16} | {'TTFT (ms)':<12} | {'Gen Tokens':<12} | {'Speed (tok/s)':<14} | {'Status':<10}")
    print("-" * 70)
    
    for approx_tok in args.steps:
        res = run_benchmark_step(args.endpoint, args.model, approx_tok, args.max_tokens)
        if res.get("error"):
            print(f"{approx_tok:<16} | {'ERR':<12} | {'0':<12} | {'0.00':<14} | {res['error'][:20]:<10}")
        else:
            print(f"{res['tokens']:<16} | {res['ttft_ms']:<12.2f} | {res['output_tokens']:<12} | {res['tok_per_sec']:<14.2f} | {'OK':<10}")

    print("=" * 70)

if __name__ == "__main__":
    main()
