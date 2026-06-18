#!/bin/bash
SK=/Users/clawdbot/.ssh/id_ed25519_spark; JUMP="<node2-host>@<NODE2_TAILNET_IP>"
ssh-add "$SK" 2>/dev/null || true
NODES="<node0-host>@<NODE0_IP> <node1-host>@<NODE1_IP> <node2-host>@<NODE2_IP>"
chk(){ local h="$1"; local cmd='if [ -f ~/nccl230-build.DONE ]; then echo DONE; grep -aE "NCCLVER|SUBNET_AWARE" ~/nccl230-build.log 2>/dev/null | tail -3; elif [ -f ~/nccl230-build.FAIL ]; then echo FAIL; tail -5 ~/nccl230-build.log; else echo RUNNING; fi'; local b=$(printf '%s' "$cmd" | base64 | tr -d '\n'); ssh -A -i "$SK" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 "$JUMP" "ssh -A -o BatchMode=yes -o ConnectTimeout=15 $h \"echo $b | base64 -d | bash\"" 2>/dev/null; }
i=0
while [ $i -lt 60 ]; do
  i=$((i+1)); done=0; fail=0; OUT=""
  for h in $NODES; do R=$(chk "$h"); OUT="$OUT\n[$h]\n$R"; echo "$R" | grep -q DONE && done=$((done+1)); echo "$R" | grep -q FAIL && fail=$((fail+1)); done
  if [ $((done+fail)) -ge 3 ]; then echo -e "RESULT done=$done fail=$fail$OUT"; break; fi
  sleep 30
done
echo "NCCL_BUILD_WATCHER_EXIT iter=$i done=$done fail=$fail"
