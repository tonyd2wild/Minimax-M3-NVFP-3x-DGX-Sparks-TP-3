#!/bin/bash
# Self-Healing Health Watchdog for MiniMax-M3 Proxy Stack
HEALTH_URL="http://127.0.0.1:8009/__health"
FAIL_COUNT=0
MAX_FAILS=3

for i in $(seq 1 $MAX_FAILS); do
  if curl -s --max-time 5 -f "$HEALTH_URL" > /dev/null 2>&1; then
    exit 0
  fi
  FAIL_COUNT=$((FAIL_COUNT + 1))
  sleep 5
done

if [ "$FAIL_COUNT" -ge "$MAX_FAILS" ]; then
  echo "$(date -u) [watchdog] Health check $HEALTH_URL failed $FAIL_COUNT times! Restarting m3-reasoning-proxy..."
  systemctl --user restart m3-reasoning-proxy.service
fi
