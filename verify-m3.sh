#!/bin/bash
# M3 TP=3 serve verification - run ON Bluey (localhost:8000). Proves generation + minimax_m3
# tool-calling work cleanly (no <mm:think> / namespace-token leaks = the sglang bug we left).
set -uo pipefail
BASE="${BASE:-http://localhost:8000}"
LEAK_RE='<mm:think>|</mm:think>|\]<\]minimax\[>\[|<tool_call>|</?invoke'

echo "===== 1. /v1/models ====="
curl -s -m 10 "$BASE/v1/models" | python3 -m json.tool 2>/dev/null | grep -E '"id"|"object"' | head

echo; echo "===== 2. plain chat (reasoning-split check) ====="
R2=$(curl -s -m 120 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "model":"minimax-m3",
  "messages":[{"role":"user","content":"In one sentence, what is the capital of France?"}],
  "max_tokens":512,"temperature":0.7}')
echo "$R2" | python3 -c 'import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]
print("content:", repr(m.get("content"))[:300])
print("reasoning_content present:", bool(m.get("reasoning_content")))
print("finish_reason:", r["choices"][0].get("finish_reason"))' 2>/dev/null || echo "PARSE FAIL: $R2" | head -c 500
echo "$R2" | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")' 2>/dev/null | grep -Eq "$LEAK_RE" && echo "LEAK in content -> FAIL" || echo "content leak-grep: PASS (no markers)"

echo; echo "===== 3. TOOL CALL (the main event) ====="
R3=$(curl -s -m 120 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "model":"minimax-m3",
  "messages":[{"role":"user","content":"What is the weather in Seattle right now? Use the tool."}],
  "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["city"]}}}],
  "tool_choice":"auto","max_tokens":1024,"temperature":0.7}')
echo "$R3" | python3 -c 'import sys,json
r=json.load(sys.stdin); m=r["choices"][0]["message"]; tc=m.get("tool_calls") or []
print("finish_reason:", r["choices"][0].get("finish_reason"))
print("num tool_calls:", len(tc))
if tc:
    f=tc[0]["function"]; print("fn name:", f.get("name"))
    try: print("ARGS OK (valid JSON):", json.loads(f.get("arguments")))
    except Exception as e: print("ARGS INVALID JSON ->", e, "raw:", f.get("arguments")[:200])
print("content:", repr(m.get("content"))[:200])' 2>/dev/null || echo "PARSE FAIL: $R3" | head -c 600
echo "$R3" | grep -Eq "$LEAK_RE" && echo "LEAK in tool response -> FAIL (parser regression)" || echo "tool-response leak-grep: PASS (no leaked tokens)"
echo; echo "===== verdict: PASS if reasoning_content present, finish_reason=tool_calls, ARGS OK, both leak-greps PASS ====="
