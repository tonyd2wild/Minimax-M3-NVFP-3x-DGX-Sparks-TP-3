#!/bin/bash
# Watches the M3 TP=3 serve bring-up on Bluey. Exits (re-invoking Kai) + pings Tony on terminal:
# SERVING (/v1/models 200) / CRASHED (container exited or EngineCore failed) / TIMEOUT.
SK=/Users/clawdbot/.ssh/id_ed25519_spark
JUMP="<node2-host>@<NODE2_TAILNET_IP>"
HEAD="<node0-host>@<NODE0_IP>"
ssh-add "$SK" 2>/dev/null || true
set -a; . /Users/clawdbot/.claude/channels/telegram/.env 2>/dev/null; set +a
TOK="${TELEGRAM_BOT_TOKEN:-$BOT_TOKEN}"; CHAT=7937060346
tg(){ curl -s "https://api.telegram.org/bot${TOK}/sendMessage" -d chat_id=$CHAT --data-urlencode "text=$1" -o /dev/null 2>/dev/null; }

REMOTE='UP=$(docker ps --format "{{.Names}}" | grep -c vllm_m3);
M=$(curl -s -m 5 http://localhost:8000/v1/models 2>/dev/null);
if echo "$M" | grep -q "minimax"; then echo "STATE=SERVING"; exit 0; fi
if [ "$UP" = "0" ]; then echo "STATE=CONTAINER_EXITED"; exit 0; fi
if docker logs --tail 60 vllm_m3 2>&1 | grep -qE "Engine core initialization failed|EngineCore failed to start"; then echo "STATE=ENGINE_FAILED"; exit 0; fi
# progress hint
P=$(docker logs --tail 40 vllm_m3 2>&1 | grep -oE "Starting to load model|Model loading took [0-9.]+|Capturing|cudagraph|init_worker|Maximum concurrency|GPU KV cache" | tail -1);
echo "STATE=WORKING ${P}"'
B64=$(printf '%s' "$REMOTE" | base64 | tr -d '\n')
probe(){ ssh -A -i "$SK" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 "$JUMP" "ssh -A -o BatchMode=yes -o ConnectTimeout=15 $HEAD \"echo $B64 | base64 -d | bash\"" 2>/dev/null; }

i=0
while [ $i -lt 120 ]; do
  i=$((i+1))
  R=$(probe)
  case "$R" in
    *STATE=SERVING*) echo "RESULT=SERVING iter=$i"; tg "🟢 M3 TP=3 endpoint is UP - /v1/models is responding on :8000. Kai's verifying generation + tool-calling now."; break ;;
    *STATE=CONTAINER_EXITED*) echo "RESULT=CONTAINER_EXITED iter=$i"; break ;;
    *STATE=ENGINE_FAILED*) echo "RESULT=ENGINE_FAILED iter=$i"; break ;;
    *) : ;;  # WORKING, keep polling
  esac
  sleep 20
done
echo "WATCHER_EXIT iter=$i last=$R"
